"""Recording and HTML visualization for an end-to-end SayCan run."""
from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image, ImageDraw


class RunVisualizer:
    """Sample simulator frames and connect them to planner/skill events."""

    def __init__(self, runtime, output, task, fps=8, cameras=("overhead",)):
        self.runtime = runtime
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.task = str(task)
        self.fps = max(1, int(fps))
        self.cameras = tuple(dict.fromkeys(cameras))
        if not self.cameras:
            raise ValueError("At least one camera is required for recording")
        self.frames = {camera: [] for camera in self.cameras}
        self.events = []
        self.stage = "initializing"
        self._last_frame_time = float("-inf")
        self._attached = True
        runtime.add_step_callback(self._on_step)

    def _annotated_frame(self, camera):
        image = Image.fromarray(self.runtime.render(camera=camera))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 52), fill=(10, 15, 24))
        draw.text((10, 7), f"MuJoCo SayCan | {self.stage}", fill=(255, 255, 255))
        draw.text(
            (10, 29),
            f"camera={camera}  sim_time={self.runtime.d.time:.2f}s",
            fill=(155, 211, 255),
        )
        return image

    def capture(self, force=False):
        now = float(self.runtime.d.time)
        if not force and now - self._last_frame_time < 1.0 / self.fps:
            return
        for camera in self.cameras:
            self.frames[camera].append(self._annotated_frame(camera))
        self._last_frame_time = now

    def _on_step(self, _runtime):
        self.capture(force=False)

    def on_stage(self, stage):
        self.stage = str(stage)
        self.events.append({
            "type": "skill_stage",
            "stage": self.stage,
            "simulation_time_s": float(self.runtime.d.time),
        })
        self.capture(force=True)

    def on_plan(self, event):
        event = json.loads(json.dumps(event, ensure_ascii=False))
        event["type"] = "planner_round"
        event["simulation_time_s"] = float(self.runtime.d.time)
        self.events.append(event)
        self.stage = (
            f"round {event['round']}: {event['selected_name']} "
            f"(Say*Can={event['score']:.3f})"
        )
        self.capture(force=True)

    @staticmethod
    def _save_gif(frames, path, fps):
        if not frames:
            return False
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=max(20, round(1000 / fps)),
            loop=0,
            optimize=False,
        )
        return True

    def _write_html(self, result, artifacts):
        rows = []
        for step in result.get("history", []):
            selected = step.get("selected_candidate_id", "-")
            for candidate in step.get("candidate_scores", []):
                chosen = "selected" if candidate.get("id") == selected else ""
                rows.append(
                    "<tr class='%s'><td>%s</td><td>%s</td><td><code>%s</code></td>"
                    "<td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%s</td></tr>"
                    % (
                        chosen,
                        step.get("round", ""),
                        html.escape(str(candidate.get("name", ""))),
                        html.escape(json.dumps(candidate.get("args", {}), ensure_ascii=False)),
                        float(candidate.get("say", 0.0)),
                        float(candidate.get("can", 0.0)),
                        float(candidate.get("score", 0.0)),
                        html.escape(str(candidate.get("rationale", ""))),
                    )
                )
        media = "".join(
            f"<figure><img src='{html.escape(name)}' alt='{html.escape(name)}'>"
            f"<figcaption>{html.escape(name)}</figcaption></figure>"
            for name in artifacts if name.endswith(".gif")
        )
        document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MuJoCo SayCan Run</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#0b1018;color:#e9f0f8}}
main{{max-width:1180px;margin:auto;padding:28px}} h1{{margin-bottom:4px}}
.status{{color:{'#77e0a0' if result.get('success') else '#ff8b8b'}}}
.media{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
figure{{margin:0;background:#151e2b;padding:10px;border-radius:10px}} img{{width:100%}}
table{{width:100%;border-collapse:collapse;background:#151e2b}}
th,td{{padding:8px;border-bottom:1px solid #2d3b4e;text-align:left;vertical-align:top}}
tr.selected{{background:#153c2b}} code{{white-space:pre-wrap;color:#b6ddff}}
</style></head><body><main>
<h1>MuJoCo SayCan</h1>
<p>{html.escape(self.task)}</p>
<h2 class="status">{html.escape(str(result.get('status', 'unknown')))}</h2>
<section class="media">{media}</section>
<h2>Say × Can 规划轨迹</h2>
<table><thead><tr><th>轮次</th><th>技能</th><th>参数</th><th>Say</th><th>Can</th><th>乘积</th><th>理由</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</main></body></html>"""
        (self.output / "report.html").write_text(document, encoding="utf-8")

    def finalize(self, result):
        self.stage = f"finished: {result.get('status', 'unknown')}"
        self.capture(force=True)
        if self._attached:
            self.runtime.remove_step_callback(self._on_step)
            self._attached = False

        artifacts = []
        for camera, frames in self.frames.items():
            gif_name = f"run_{camera}.gif"
            if self._save_gif(frames, self.output / gif_name, self.fps):
                artifacts.append(gif_name)
                png_name = f"final_{camera}.png"
                frames[-1].save(self.output / png_name)
                artifacts.append(png_name)
        (self.output / "timeline.json").write_text(
            json.dumps(self.events, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts.append("timeline.json")
        self._write_html(result, artifacts)
        artifacts.append("report.html")
        return {
            "output_directory": str(self.output.resolve()),
            "fps": self.fps,
            "cameras": list(self.cameras),
            "artifacts": artifacts,
        }

