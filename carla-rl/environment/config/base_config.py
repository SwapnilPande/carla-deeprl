import pickle


class BaseConfig:
    def __init__(self):
        pass

    def verify(self, ignore_keys = []):
        # Ignore keys contains keys we want to skip during verification

        parameters = vars(self)

        for name, value in parameters.items():
            # Check that value is not None
            # Raise Exception if value is None
            if (name not in ignore_keys):
                if(value is None):
                    import ipdb; ipdb.set_trace()
                    raise Exception("Missing value for parameter {} in config {}".format(
                            name,
                            self.__class__.__name__
                    ))
                # If object is another config object, call verify on that config as well
                if isinstance(value, BaseConfig):
                    value.verify()

        print("Verified config {}. Note: this just checks for missing values!".format(self.__class__.__name__))

    def set_parameter(self, name, value, force = False):
        """ Set the value of a parameter in the config.

        Set the parameter of a config. NOTE: this will raise an exception if the parameter is not specified in the config.
        This behavior is to prevent accidentally setting a non-existent parameter due to a typo.
        """

        if(hasattr(self, name) or force):
            setattr(self, name, value)
        else:
            raise Exception("Parameter {} not found in config {}".format(name, self.__class__.__name__))

    def get_parameter(self, name):
        """ Get the value of a parameter. This is equivalent to config.{PARAM_NAME}
        """

        return getattr(self, name)