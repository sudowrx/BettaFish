"""
生成 AI 舆情深度研究报告（完整 5 段流程 + 最终报告）

用法:
    cd ~/projects/BettaFish/InsightEngine
    python3 tools/gen_report.py [关键词]

默认关键词 "人工智能"。生成完成后自动更新 reports/LATEST.md 指针文件。

配套核对:
    python3 ~/.hermes/skills/devops/bettafish-report-hallucination/scripts/verify_report_numbers.py \
        reports/deep_search_report_<时间戳>.md
"""
import os
import sys
import time
import glob

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(TOOLS_DIR)          # .../BettaFish/InsightEngine
PROJECT_ROOT = os.path.dirname(ENGINE_DIR)       # .../BettaFish

# `from InsightEngine.agent import ...` 需要项目根在 sys.path 里
# （BettaFish/），不是 InsightEngine/ —— 插错层级会 ImportError
sys.path.insert(0, PROJECT_ROOT)
# 工作目录设为 InsightEngine/，这样 reports/ 相对路径才对
os.chdir(ENGINE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("/tmp/gen_report.log", level="DEBUG")

from InsightEngine.agent import DeepSearchAgent
from InsightEngine.utils.config import settings

# 完整流程：所有段落，1 次反思（控制总时长）
settings.MAX_REFLECTIONS = 1

QUERY = sys.argv[1] if len(sys.argv) > 1 else "人工智能"


def update_latest_pointer(report_path: str) -> str:
    """写 reports/LATEST.md 指针，指向最新报告。

    指针文件的存在意义：reports/ 下文件名带时间戳，靠 `ls -t` 取最新很容易
    误取到"篇幅最长但有失真"的旧版。LATEST.md 是唯一权威入口。
    """
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(report_path)))
    latest_md = os.path.join(reports_dir, "LATEST.md")
    rel = os.path.basename(report_path)
    size = os.path.getsize(report_path)
    # 用报告文件的修改时间，不是当前时间 —— 手工补写指针时当前时间是错的
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(report_path)))
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(
            f"# 最新舆情报告\n\n"
            f"- **文件**: [`{rel}`](./{rel})\n"
            f"- **生成时间**: {mtime}\n"
            f"- **大小**: {size} 字节\n"
            f"- **关键词**: {QUERY}\n\n"
            f"> 本文件由 `tools/gen_report.py` 自动生成，每次跑完报告都会覆盖。\n"
            f"> 手工报告请勿依赖 `ls -t` 取最新 —— 旧版本可能存在已修正的数字失真。\n"
        )
    return latest_md


async def main():
    t0 = time.time()
    agent = DeepSearchAgent()
    print(f"[GEN] agent 初始化 {time.time() - t0:.1f}s", flush=True)

    agent._generate_report_structure(QUERY)
    n = len(agent.state.paragraphs)
    print(f"[GEN] 报告结构 {time.time() - t0:.1f}s, 段落数={n}", flush=True)

    for i in range(n):
        t = time.time()
        print(f"\n[GEN] === 段落 {i + 1}/{n} 搜索 ===", flush=True)
        agent._initial_search_and_summary(i)
        print(f"[GEN] 段落 {i + 1} 初始完成 {time.time() - t:.1f}s", flush=True)

        t = time.time()
        print(f"[GEN] === 段落 {i + 1} 反思 ===", flush=True)
        agent._reflection_loop(i)
        s = agent.state.paragraphs[i].research.latest_summary or ""
        print(f"[GEN] 段落 {i + 1} 反思完成 {time.time() - t:.1f}s, 总结长度={len(s)}", flush=True)

    print("\n[GEN] === 生成最终报告 ===", flush=True)
    t = time.time()
    final = agent._generate_final_report()
    print(f"[GEN] 最终报告 {time.time() - t:.1f}s, 长度={len(final)}", flush=True)

    agent._save_report(final)
    print("[GEN] 已保存到 reports/ 目录", flush=True)

    # 找刚生成的文件（_save_report 用时间戳命名，跨秒时用 mtime 兜底）
    candidates = sorted(
        glob.glob(os.path.join("reports", "deep_search_report_*.md")),
        key=os.path.getmtime,
    )
    if candidates:
        latest = candidates[-1]
        pointer = update_latest_pointer(latest)
        print(f"[GEN] 报告文件: {latest}", flush=True)
        print(f"[GEN] 指针已更新: {pointer}", flush=True)

    print(f"\n[GEN] 总耗时 {time.time() - t0:.1f}s", flush=True)
    print("[GEN] DONE", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
