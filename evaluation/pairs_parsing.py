def lfw_pairs(path):
    """
    Reads the lfw pairs.txt and generates filenames from it
    :param path:
    :return:
    """
    with open(path, 'r') as file:
        lines = file.readlines()