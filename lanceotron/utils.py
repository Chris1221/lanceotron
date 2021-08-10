def make_directory_name(directory: str) -> str:
    """Ensures a trailing slash on directory names.

    Args:
        directory (str): Path to the directory

    Returns:
        str: Directory name with a guarenteed trailing slash
    """
    if directory[-1] != "/":
        directory += "/"

    return directory