import shutil
from dataclasses import asdict

import numpy as np

from tests.helpers.builders import EmbeddingsJobConfigurationBuilder
from text2sql.commons.io_utils import (
    load_configuration_to_target_dataclass,
    read_json,
    save_configuration_to_yaml,
    save_embeddings,
)
from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfigurationDto
from text2sql.modules.schema_linking.domain.model import Embedding
from text2sql.settings import TESTS_DIR

TMP_OUTPUT_DIR = TESTS_DIR.joinpath("resources", "temp_output_dir")


class TestExperimentArtifactsStoringUtilities:

    def teardown_method(self):
        if TMP_OUTPUT_DIR.exists():
            shutil.rmtree(TMP_OUTPUT_DIR.path)

    def test_experiment_configuration_saving(self) -> None:
        # GIVEN
        original_experiment_config = EmbeddingsJobConfigurationBuilder.default().build()

        # AND
        path_to_stored_config_file_yaml = TMP_OUTPUT_DIR.joinpath("config.yaml")
        save_configuration_to_yaml(
            filepath=path_to_stored_config_file_yaml, config_dict=asdict(original_experiment_config)
        )

        # WHEN
        restored_experiment_config = load_configuration_to_target_dataclass(
            path_to_stored_config_file_yaml, target_dataclass=EmbeddingsJobConfigurationDto
        )

        assert restored_experiment_config.to_domain() == original_experiment_config

    def test_embeddings_saving(self) -> None:
        # GIVEN
        original_embeddings = [
            Embedding(example_id=0, embedding=np.array([0.112, 0.68, 1.14777, 6.75])),
            Embedding(example_id=1, embedding=np.array([0.189, 55555.6, 19.29, 0.00004])),
        ]

        # AND
        path_to_stored_embeddings = TMP_OUTPUT_DIR.joinpath("embeddings.json")
        save_embeddings(original_embeddings, path_to_stored_embeddings)

        # WHEN
        restored_embeddings = [Embedding(**embedding) for embedding in read_json(path_to_stored_embeddings)]

        # THEN
        assert restored_embeddings[0].example_id == original_embeddings[0].example_id
        assert np.array_equal(restored_embeddings[0].embedding, original_embeddings[0].embedding)
        assert restored_embeddings[1].example_id == original_embeddings[1].example_id
        assert np.array_equal(restored_embeddings[1].embedding, original_embeddings[1].embedding)
