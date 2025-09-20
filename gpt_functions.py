import datetime
import zoneinfo

def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

tools = [
    {
        "name": "get_current_time",
        "description": "지정한 타임존의 현재 시간을 문자열로 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "예: Asia/Seoul, UTC, America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        },
    }
]