

class Tools:
    def __init__(self):
        pass

    def read_file(self, path2data_file):
        strings = []
        file_object = open(path2data_file, "r")
        for line in file_object.readlines():
            line = line.strip()
            strings.append(line)
        return strings
