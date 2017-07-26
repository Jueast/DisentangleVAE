
class Dataset:
    def __init__(self):
        self.name = "abstract"
        self.data_dims = []
        self.width = -1
        self.height = -1
        self.range = [0.0, 1.0]
        self._epoch = 0
    """ Get next training batch """
    def next_batch(self, batch_size):
        self.handle_unsupported_op()
        return None

    def __len__(self):
        self.handle_unsupported_op()
        return None
    
    def epoch(self):
        self.handle_unsupported_op()
        return None

    def display(self, image):
        return image

    def handle_unsupported_op(self):
        print("Unsupported Operation")
        raise(Exception("Unsupported Operation"))


