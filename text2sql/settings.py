from upath import UPath

PROJECT_PATH = UPath(__file__).parent.parent.resolve()

RESOURCES_DIR = PROJECT_PATH.joinpath("resources")
CACHE_DIR = PROJECT_PATH.joinpath(".cache")
CONFIG_DIR = PROJECT_PATH.joinpath("configs")

DATASETS_DIR = RESOURCES_DIR.joinpath("datasets")
PREDICTIONS_DIR = RESOURCES_DIR.joinpath("predictions")
ASSETS_DIR = RESOURCES_DIR.joinpath("assets")

TESTS_DIR = PROJECT_PATH.joinpath("tests")
