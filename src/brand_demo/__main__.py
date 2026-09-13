"""Interactive brand demo. A plan/offline run never loads API credentials."""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bias_eval.logging import configure_logging, sanitize_for_log

from .clients import FixtureClient, LiveClient
from .core import plan, render
from .workflow import run


def load_keys(path: Path = Path(".env")) -> None:
    """Read only two literal values; never execute shell expressions from .env."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, separator, value = line.removeprefix("export ").partition("=")
            key = key.strip()
            if (
                separator
                and key in {"MINIMAX_API_KEY", "SERPER_API_KEY"}
                and not os.environ.get(key)
            ):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ[key] = value
    missing = [
        k for k in ("MINIMAX_API_KEY", "SERPER_API_KEY") if not os.environ.get(k, "").strip()
    ]
    if missing:
        raise ValueError("Eksik anahtarlar: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand")
    parser.add_argument("--sector", help="vpn, kozmetik veya herhangi bir sektör")
    parser.add_argument("--competitor", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Gerçek, ücret doğurabilecek çağrılar")
    mode.add_argument("--offline", action="store_true", help="Açıkça sentetik, sıfır API demo")
    parser.add_argument(
        "--yes", action="store_true", help="Canlı çağrı onayını atla (maliyeti kabul et)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Hatalı/belirsiz çağrıları bir kez daha dene; tekrar ücretlenebilir",
    )
    parser.add_argument("--run-id", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--output", type=Path, default=Path("data/processed/brand_demo"))
    parser.add_argument("--no-local-model", action="store_true")
    parser.add_argument(
        "--rebuild-report", type=Path, help="Mevcut rapordan API kullanmadan aksiyon planı oluştur"
    )
    args = parser.parse_args(argv)
    try:
        if args.rebuild_report:
            if args.live or args.offline or args.retry_failed:
                parser.error("--rebuild-report canlı/offline toplama bayraklarıyla birleştirilemez")
            from .actions import rebuild

            result, destination = rebuild(args.rebuild_report)
            print(render(result))
            print("API çağrısı: 0. Aksiyon raporu:", destination)
            return 0
        if not args.brand or not args.sector:
            if not sys.stdin.isatty():
                parser.error(
                    "--brand ve --sector gerekli; interaktif kullanım için make brand-demo"
                )
            args.brand = args.brand or input("Markanız: ").strip()
            args.sector = args.sector or input("Sektörünüz (örn. vpn, kozmetik, kahve): ").strip()
        config = plan(args.brand, args.sector, args.competitor, args.run_id, args.offline)
        print(
            json.dumps(
                {
                    "brand": config["brand"],
                    "sector": config["sector"],
                    "queries": config["queries"],
                    "run_id": config["run_id"],
                    "mode": config["mode"] if args.live or args.offline else "plan_only",
                    "normal_uncached_calls": (
                        config["normal_uncached_calls"]
                        if not args.offline
                        else {"minimax": 0, "serper": 0}
                    ),
                    "maximum_requested_output_tokens": (
                        config["maximum_requested_output_tokens"] if not args.offline else 0
                    ),
                    "output": str(args.output / config["id"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not args.live and not args.offline:
            print("Yalnız plan; API çağrısı yok. --offline veya onaylı --live ile çalıştırın.")
            return 0
        if args.live:
            print(
                "Ücretsiz kota garantisi yok. Anahtarlar yalnız MiniMax ve Serper'a gönderilir; sorgular ve snippet'ler bu sağlayıcılarda işlenir. Otomatik tekrar yok."
            )
            if args.retry_failed:
                print(
                    "UYARI: Hatalı/belirsiz adımlar yeniden ücretlenebilir; her adım toplam en fazla 3 kez denenebilir."
                )
            if not args.yes and (
                not sys.stdin.isatty()
                or input("Başlatılsın mı? [evet/Hayır]: ").strip().casefold() != "evet"
            ):
                print("İptal edildi; API çağrısı yapılmadı.")
                return 0
            load_keys()
        configure_logging(log_file=args.output / config["id"] / "demo.log")

        async def execute() -> dict:
            if args.offline:
                return await run(
                    config, args.output, FixtureClient(config["brand"]), use_local=False
                )
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120, connect=20), follow_redirects=False
            ) as http:
                return await run(
                    config,
                    args.output,
                    LiveClient(http),
                    retry_failed=args.retry_failed,
                    use_local=not args.no_local_model,
                )

        result = asyncio.run(execute())
        print(render(result))
        print("Rapor:", args.output / config["id"] / "report.md")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print("Demo durdu: " + sanitize_for_log(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print(
            "Durduruldu. Tamamlanan adımlar korundu; pending istek ücretlenmiş olabilir.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
