"""H3 LoRAs: a short curated SFW list (Hugging Face, pinned) plus the owner's own files in Drive.

Curated files ride along as extra weight files (Drive cache hit after a CPU prep, else Hugging Face).
Own files sit in Drive ``AI-Studio/<owner>/loras/h3`` and are fetched by the runtime at job start.
They are chained as LoraLoaderModelOnly right after the UNET loader, so the Turbo LoRA and the
sigma-shift node keep their place after them.
"""

from __future__ import annotations

import copy

MAX_LORAS = 3
MAX_BYTES = 1_500_000_000
H3_WORKFLOWS = {"minimax-h3-i2v", "minimax-h3-ref2v"}

CURATED = [
    {
        "id": "anime-slider-ref2v",
        "label": "แถบเลื่อนล็อกสไตล์ 2D อนิเมะ/CG (ป้องกันภาพกลายเป็น 3D หรือคนจริง)",
        "note": "ควบคุมสไตล์โดยรวมให้เอนเอียงไปทาง 2D อนิเมะ/CG หรือภาพสมจริง ป้องกันไม่ให้ภาพอ้างอิงอนิเมะ CG กลายเป็น 3D/คนจริงเมื่อเคลื่อนไหว ไม่ต้องใช้ Trigger Word แนะนำความแรง 1.0–3.0 (ค่าเริ่มต้น 1.5)",
        "strength": 1.5,
        "workflows": ["minimax-h3-ref2v"],
        "url": "https://huggingface.co/RunningHubAI/rh-minmax-h3-2d-lora",
        "repo_id": "RunningHubAI/rh-minmax-h3-2d-lora",
        "filename": "H3_Ref2V_Anime_Slider_v1.safetensors",
        "revision": "3568edd3d16b8d3c637814b55f628ec4c5cbd121",
        "sha256": "60b0e8523c72a94eb84dca63eaf88bb0f0b844f053c86d8f318c4f8b7d4eac11",
        "size": 77580464,
    },
    {
        "id": "live-wallpaper-r64",
        "label": "วอลเปเปอร์เคลื่อนไหว CG R64 (มุมกล้อง + การเคลื่อนไหวละเอียด)",
        "note": "เหมาะสำหรับภาพคอสตูม/CG ที่มีการเคลื่อนไหวเล็กน้อย (เส้นผม, ชายกระโปรง, แสง, การหายใจ) และมุมกล้องตามพรอมต์ แนะนำให้พรอมต์ขึ้นต้นด้วย “live_wallpaper: Animate <Picture 1> with ...”; แนะนำความแรง 0.8–1.0",
        "strength": 1.0,
        "workflows": ["minimax-h3-ref2v"],
        "url": "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI",
        "repo_id": "Alissonerdx/Minimax-H3-ComfyUI",
        "filename": "loras/minimax_h3_live_wallpaper_v1.0_ref2va_r64.safetensors",
        "revision": "4bb14d4a611179d4cf633e74d3ae33e965ae2301",
        "sha256": "7ea5e75ae1514583a714e10caa7b215c04061c0dc27594284a9d8274b3a53348",
        "size": 620285440,
    },
    {
        "id": "live-wallpaper-r32",
        "label": "วอลเปเปอร์เคลื่อนไหว CG R32 (รักษาใบหน้าได้นิ่งกว่า)",
        "note": "เหมาะสำหรับการเคลื่อนไหว CG ภาพเดี่ยวหรือหลายภาพ รักษาหน้าตาได้นิ่งกว่า แนะนำให้พรอมต์ขึ้นต้นด้วย “live_wallpaper: Animate <Picture 1> with ...”; แนะนำความแรง 0.8–1.0",
        "strength": 0.9,
        "workflows": ["minimax-h3-ref2v"],
        "url": "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI",
        "repo_id": "Alissonerdx/Minimax-H3-ComfyUI",
        "filename": "loras/minimax_h3_live_wallpaper_v1.0_ref2va_r32.safetensors",
        "revision": "4bb14d4a611179d4cf633e74d3ae33e965ae2301",
        "sha256": "3c9ff5fe4d9adc151d7b52c67fb79250c0012376061b8d44d01ba19ad824c43a",
        "size": 310168936,
    },
    {
        "id": "anti-burn-v1",
        "label": "Anti-Burn (ป้องกันหน้าไหม้/เส้นแตกเมื่อต่อวิดีโอหลายตอน)",
        "note": "ใช้เมื่อนำเฟรมสุดท้ายของวิดีโอก่อนหน้ามาต่อตอนที่ 2, 3, 4 เพื่อป้องกันไม่ให้ใบหน้าและเส้นสายคมจัดจนไหม้เกรียม ห้ามใส่ในตอนแรก! แนะนำความแรง 0.5",
        "strength": 0.5,
        "workflows": ["minimax-h3-i2v", "minimax-h3-ref2v"],
        "url": "https://huggingface.co/Imagineer13/H3-Anti-Burn-LoRA",
        "repo_id": "Imagineer13/H3-Anti-Burn-LoRA",
        "filename": "h3_antiburn_universal_v1.safetensors",
        "revision": "b98eb1d3f5574d1b8f57d6b0e720d2f047cfadb3",
        "sha256": "8f76dc3c63afa8c556c620b4b7f2ffeeb666f0aa9382a9ce3a07fd6120d0f154",
        "size": 310168728,
    },
    {
        "id": "better-human-motion",
        "label": "Better Human Motion (การเคลื่อนไหวของมนุษย์ที่เป็นธรรมชาติยิ่งขึ้น)",
        "note": "ท่าทางและสรีระของตัวละครต่อเนื่องและเป็นธรรมชาติยิ่งขึ้น ผู้สร้างแนะนำ 0.4–0.8 ควรเขียนพรอมต์สั้นกระชับ ภาพอนิเมะ CG ควรระวังการซ้อนทับ",
        "strength": 0.6,
        "workflows": ["minimax-h3-i2v", "minimax-h3-ref2v"],
        "url": "https://huggingface.co/vpakarinen/better-human-motion-h3-lora",
        "repo_id": "vpakarinen/better-human-motion-h3-lora",
        "filename": "better_motion_h3_lora_v1_500.safetensors",
        "revision": "df545ebfbdaf30799196098f6df157ee4388b766",
        "sha256": "285f9ed3fbb735c3625fca0d22166c7450ad7acb552a869fcbbd8e86e79d1f4c",
        "size": 310168816,
    },
    {
        "id": "motion-repair-v2",
        "label": "Motion Continuity Repair V2 (ป้องกันการเคลื่อนไหวขาดตอน)",
        "note": "ซ่อมแซมการเคลื่อนไหวใหญ่ เช่น หมุนตัว ตีลังกา กระโดด ไม่ให้ขาดตอนหรือช้าผิดปกติ แนะนำให้ใช้เดี่ยวๆ ที่ประมาณ 0.9 อย่าผสมกับ LoRA การเคลื่อนไหวตัวอื่น",
        "strength": 0.9,
        "workflows": ["minimax-h3-i2v", "minimax-h3-ref2v"],
        "url": "https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair",
        "repo_id": "JOKER141/MiniMax-H3-General-Motion-Continuity-Repair",
        "filename": "Motion_Repair_V2.safetensors",
        "revision": "1676b352bb9f62407f76aee2ad55db2f36d04471",
        "sha256": "edfa0e2858caeba7be9108883f61da8525d3c58af998c94ac7a449d9603d7574",
        "size": 155110272,
    },
    {
        "id": "camera-motion",
        "label": "Camera Motion (การควบคุมมุมกล้อง)",
        "note": "มุมกล้องซูมเข้า ซูมออก แพนรอบ ทรงตัวแบบมือถือ พรอมต์ต้องขึ้นต้นด้วย “camera motion,” ผู้สร้างแนะนำ 0.8–1.0 หากเกิน 1.2 ภาพจะไม่นิ่ง",
        "strength": 0.8,
        "workflows": ["minimax-h3-i2v", "minimax-h3-ref2v"],
        "url": "https://huggingface.co/Jojocodex/minimax-h3-Camera-Motion-lora",
        "repo_id": "Jojocodex/minimax-h3-Camera-Motion-lora",
        "filename": "camera_motion_h3_lora_v1_1000_pruned.safetensors",
        "revision": "fac88d6196ca84f3cee014ec79c71dc73572ab0a",
        "sha256": "c126738c887804ace3a4be4a3156fcca70517613c7a7ece4e5938d372419186b",
        "size": 155109488,
    },
    {
        "id": "anime-motion",
        "label": "Anime Motion (การเคลื่อนไหวสไตล์อนิเมะ เฉพาะแปลงภาพเป็นวิดีโอ)",
        "note": "การเคลื่อนไหววนลูปสไตล์อนิเมะ: กะพริบตา สีหน้า เส้นผมและฉากหลังขยับเบาๆ จะทำให้ท่าทางใหญ่ลดลง Trigger Word “Anime-Motion” แนะนำเริ่มทดลองที่ 0.8 ใช้ได้เฉพาะ I2V",
        "strength": 0.8,
        "workflows": ["minimax-h3-i2v"],
        "url": "https://huggingface.co/prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA",
        "repo_id": "prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA",
        "filename": "MiniMax-H3-I2V-Anime-Motion-LoRA-1400.safetensors",
        "revision": "ca20427ab7488b80fc3a14363561431e27a8f974",
        "sha256": "28e36c2e62b464b1779fc540ad2dcd6a0ed8a71ea20c6b0ca78745f7f440a2b7",
        "size": 74570616,
    },
]

_PUBLIC = ("id", "label", "note", "strength", "workflows", "url", "size")


def curated_for(workflow_key: str | None) -> list[dict]:
    return [{k: c[k] for k in _PUBLIC} for c in CURATED if workflow_key in c["workflows"]]


def curated_weight(entry: dict) -> dict:
    from pathlib import PurePosixPath

    name = PurePosixPath(entry["filename"]).name
    return {
        "repo_id": entry["repo_id"],
        "filename": entry["filename"],
        "revision": entry["revision"],
        "sha256": entry["sha256"],
        "size": entry["size"],
        "target": f"loras/curated/{name}",
        "lora": True,
    }


def is_lora_file(name: str, size: int | None) -> bool:
    return name.lower().endswith(".safetensors") and size is not None and 0 < int(size) <= MAX_BYTES


def insert_loras(graph: dict, chain: list[tuple[str, float]]) -> dict:
    """UNET -> LoRA 1 -> LoRA 2 -> ... -> (whatever consumed the UNET before: Turbo LoRA, shift, scheduler...)."""
    graph = copy.deepcopy(graph)
    unet = next(k for k, n in graph.items() if n.get("class_type") == "UNETLoader")
    consumers = [(n, inp) for n in graph.values() for inp, v in n["inputs"].items() if v == [unet, 0]]
    source = [unet, 0]
    for i, (name, strength) in enumerate(chain):
        node = str(900 + i)
        graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": source, "lora_name": name, "strength_model": strength}}
        source = [node, 0]
    for n, inp in consumers:
        n["inputs"][inp] = source
    return graph
