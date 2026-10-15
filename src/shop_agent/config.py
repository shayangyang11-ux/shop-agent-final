"""配置层。

12-Factor 原则：配置进环境，代码与配置分离。
读取优先级：真实环境变量 > .env 文件 > 代码内默认值。

纪律（大厂红线，面试真会问）：
- .env 永不进 git（见 .gitignore），仓库里只放 .env.example
- 任何代码、日志、异常信息里不允许出现 api_key 明文
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。所有项带 SHOP_AGENT_ 前缀，避免污染宿主环境变量。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHOP_AGENT_",
        extra="ignore",
    )

    # ---- LLM（OpenAI 兼容协议；默认指向智谱 GLM，可一行切换供应商）----
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_api_key: str = ""
    llm_model: str = "glm-4.6"  # 按你账号可用的最新型号调整
    llm_temperature: float = 0.3  # 结构化生成要的是稳定，不是发散
    llm_max_tokens: int = 4096
    # 单次请求超时（秒）。页面生成是长任务（单次最多 4096 tokens 的
    # 结构化 JSON），服务端高峰期可能超过 60 秒 —— 180 秒起步。
    llm_timeout: float = 180.0
    llm_struct_retries: int = 2  # JSON 校验失败后的带错重试次数（批次2 用到）

    # ---- 供应商侧故障防御（节流 / SDK 重试 / 应用层长退避）----
    # 第一层：请求节流。任意两次 LLM 请求的启动间隔下限（秒）。
    # 防的是 RPM 类限流：并发再低，不隔间距地连发照样撞墙。
    # 免费档账号建议 12（≈每分钟 5 次，多数档位的安全线）。
    llm_min_request_interval: float = 5.0
    # 第二层：openai SDK 内建重试（秒级指数退避，抹平瞬时抖动与偶发超时）
    llm_max_retries: int = 3
    # 第三层：SDK 重试耗尽后的应用层长退避 —— 429 等配额窗口翻页，
    # 超时/断连给服务端喘息；耗尽后节点层退化处理（占位/错误路由）
    llm_rate_backoff_retries: int = 2
    # 长退避的基础等待秒数（指数递增：base、2×base、4×base）
    llm_rate_backoff_base: float = 20.0

    # ---- 并行度（Send fan-out 的并发上限，批次4 用到）----
    # 3 而不是更高：免费/低档位账号的限流按"同时在飞的请求数"计，
    # 并发越高越容易整批撞 429。赶时间可 .env 里调回。
    max_parallel_pages: int = 3

    # ---- 持久化（checkpointer 落盘位置，批次5 用到）----
    db_path: Path = Path(".shop_agent/checkpoints.sqlite")

    @property
    def is_configured(self) -> bool:
        """是否已配置可用的 API Key。CLI 启动时先检查，给出可读错误。"""
        return bool(self.llm_api_key)

    def ensure_runtime_dirs(self) -> None:
        """保证 checkpointer 数据库的父目录存在。幂等，可反复调用。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """进程级单例。测试里需要重置时用 get_settings.cache_clear()。"""
    return Settings()
