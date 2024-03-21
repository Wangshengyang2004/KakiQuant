from omegaconf import DictConfig, OmegaConf
import hydra
from kaki.utils.check_root_base import find_and_add_project_root

@hydra.main(version_base=None, config_path=f"{find_and_add_project_root()}/config/", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()