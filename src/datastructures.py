
class DataSample:
    """ Represents a single data sample
    """

    def __init__(self, path: str, label: str):
        self.path = path.strip()
        self.label = label.strip()

    def get_path(self) -> str:
        return self.path

    def get_label(self) -> str:
        return self.label

