import typing

import ConfigSpace
import pytest
from src.dehb.optimizers.de import DEBase

def create_toy_DEBase(configspace: ConfigSpace.ConfigurationSpace):
    """Creates a toy DEBase instance for conversion tests.

    Args:
        configspace (ConfigurationSpace): Searchspace to use

    Returns:
        DEBase: DEBase object for testing
    """
    dim = len(configspace.get_hyperparameters())
    return DEBase(f=lambda: 1, cs=configspace, dimensions=dim, pop_size=10, max_age=5,
                  mutation_factor=0.5, crossover_prob=0.5, strategy="rand1_bin", budget=1)

class TestConversion():
    """Class that bundles all ConfigSpace/vector conversion tests.

    These tests include conversion tests for constant, categorical, ordinal,
    float and integer hyperparameters.
    """
    def test_constant(self):
        """Test for constant hyperparameter."""
        cs = ConfigSpace.ConfigurationSpace(
            space={
                "test_const": "constant",
            },
        )

        de = create_toy_DEBase(cs)

        test_config = cs.sample_configuration()

        vector = de.configspace_to_vector(test_config)
        assert vector[0] == 0

        converted_conf = de.vector_to_configspace(vector)
        assert converted_conf == test_config

    def test_categorical(self):
        """Test for categorical hyperparameter."""
        cs = ConfigSpace.ConfigurationSpace(
            space={
                "test_categorical": ["a", "b", "c"],
            },
        )

        de = create_toy_DEBase(cs)

        test_config = cs.sample_configuration()

        vector = de.configspace_to_vector(test_config)

        assert vector[0] <= 1
        assert vector[0] >= 0

        converted_conf = de.vector_to_configspace(vector)
        assert converted_conf == test_config

    def test_ordinal(self):
        """Test for ordinal hyperparameter."""
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("test_ordinal", sequence=[10, 20, 30]))

        de = create_toy_DEBase(cs)

        test_config = cs.sample_configuration()

        vector = de.configspace_to_vector(test_config)

        assert vector[0] <= 1
        assert vector[0] >= 0

        converted_conf = de.vector_to_configspace(vector)
        assert converted_conf == test_config

    def test_float(self):
        """Test for float hyperparameter."""
        cs = ConfigSpace.ConfigurationSpace(
            space={
                "test_float": (1.0, 10.0),
            },
        )

        de = create_toy_DEBase(cs)

        test_config = cs.sample_configuration()

        vector = de.configspace_to_vector(test_config)

        assert vector[0] <= 1
        assert vector[0] >= 0

        converted_conf = de.vector_to_configspace(vector)
        assert converted_conf == test_config

    def test_integer(self):
        """Test for integer hyperparameter."""
        cs = ConfigSpace.ConfigurationSpace(
            space={
                "test_int": (1, 10),
            },
        )

        de = create_toy_DEBase(cs)

        test_config = cs.sample_configuration()

        vector = de.configspace_to_vector(test_config)

        assert vector[0] <= 1
        assert vector[0] >= 0

        converted_conf = de.vector_to_configspace(vector)
        assert converted_conf == test_config