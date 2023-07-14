import os

from transformers.configuration_utils import cached_file, is_remote_url
from transformers.utils import CONFIG_NAME


def check_huggingface_model_files_are_local(pretrained_model_name_or_path: str, **kwargs):
    # Check to see if the model files are local
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", "")
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    commit_hash = kwargs.pop("_commit_hash", None)

    user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
        # Special case when pretrained_model_name_or_path is a local file
        resolved_config_file = pretrained_model_name_or_path
        return True
    elif is_remote_url(pretrained_model_name_or_path):
        return False
    else:
        configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)

        try:
            # Load from local folder or from cache or download from model Hub and cache
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                configuration_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=True,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )
            if resolved_config_file is None:
                return False
            if os.path.isfile(resolved_config_file):
                return True
            return False
        except Exception:
            return False
