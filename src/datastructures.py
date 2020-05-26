
class DataSample:
    """ Represents a single data sample
    """

    def __init__(self, path: str, label):
        self.path = path.strip()
        self.label = label.strip() if type(label) == str else label

    def get_path(self) -> str:
        return self.path

    def get_label(self):
        return self.label

