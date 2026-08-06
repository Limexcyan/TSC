from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

class SettingsBase(BaseSettings):
    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="TSC_",
        extra="ignore",
    )