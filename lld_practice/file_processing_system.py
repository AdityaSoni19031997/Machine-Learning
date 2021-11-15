import heapq
from dataclasses import dataclass, field
from operator import lt
from typing import Dict, List, Optional, Tuple

# default collection name if none is specified.
DEFAULT_COLLECTION_NAME = "default_collection"

"""
Time Taken By Me -> 33 mins 18 secs

Atlassian LLD Round -:

    Design the following -:
        Given a list of [FileName, FileSize, [Collection]]
        - A collection can have 1 or more files.
        - Same file can be a part of more than 1 collection.

    How would you design a system
    - To calculate total size of files processed.
    - To calculate Top-K collections based on size.

    Example:
        file1.txt(size: 100)
        file2.txt(size: 200) in collection "collection1"
        file3.txt(size: 200) in collection "collection1"
        file4.txt(size: 300) in collection "collection2"
        file5.txt(size: 100)

    Output:

        Total size of files processed: 900
        Top 2 collections:
            - collection1 : 400
            - collection2 : 300
"""


@dataclass
class Attributes:
    # dummy base class which can store some common attributes b/w File and Directory.
    pass


@dataclass()
class File(Attributes):
    # This represents a file in our file system.
    name: str
    size: float
    dir_name: str


@dataclass
class Directory(Attributes):
    # This represents a directory in our file system.
    name: str
    size: float = 0
    files: List[File] = field(default_factory=list)


class DirectoryWithSize(object):
    def __init__(self, dir_name:str, dir_size:float) -> None:
        self.dir_name = dir_name
        self.dir_size = dir_size

    def __lt__(self, other):
        return lt(self.dir_size, other.dir_size)


@dataclass
class FileSystem:
    # This is the file system that we are trying to model here

    _total_file_system_size: float = 0
    all_files: Dict[str, float] = field(default_factory=dict)
    directory_mapping: Dict[str, Directory] = field(default_factory=dict)
    directory_present_in_system: set = field(default_factory=set)

    def get_total_file_system_size(self) -> float:
        return self._total_file_system_size

    def add_file_to_directory(
        self, file_name: str, file_size: float, file_directory: Optional[str]
    ) -> None:

        # add the directory to our file system first if it doesn't exists.
        if file_directory not in self.directory_present_in_system:
            file_directory = file_directory or DEFAULT_COLLECTION_NAME
            self.directory_present_in_system.add(file_directory)
            self.directory_mapping[file_directory] = Directory(name=file_directory)

        # create the file object and update the respective collections accordingly.
        current_file = File(
            name=file_name,
            size=file_size,
            dir_name=file_directory,
        )
        current_directory = self.directory_mapping.get(file_directory)
        current_directory.files.append(current_file)
        current_directory.size += file_size

        # increment the global file system size
        self._total_file_system_size += file_size
        self.all_files[current_file.dir_name] = current_directory.size

        print(
            f"File named {file_name} and size {file_size} was successfully added to our file_system under {file_directory}."
        )

    def get_top_k_directory(self, top_k: int) -> List[Tuple[str, float]]:
        # let's make a heap from the lsit of <dir_name, dir_size> and then get the top_k basically.
        # it can actually be moved out and we can maintain a fixed heap in global space as well.

        _max_heap = []

        for dir_name, dir_size in self.all_files.items():
            heapq.heappush(_max_heap, DirectoryWithSize(dir_name, -1 * dir_size))

        _results = []
        for _ in range(0, top_k):
            dir_obj = heapq.heappop(_max_heap)
            dir_name, dir_size = dir_obj.dir_name, -1 * dir_obj.dir_size
            _results.append((dir_name, dir_size))
        return _results


if __name__ == "__main__":
    files = [
        ["file_1.txt", 10000, ""],
        ["file_2.txt", 1000, "collection_1"],
        ["file_3.txt", 1210, "collection_2"],
        ["file_4.txt", 300, "collection_1"],
        ["file_5.txt", 600, "collection_2"],
        ["file_6.txt", 500, "collection_5"],
    ]
    top_k = 2

    fp = FileSystem()

    for (file_name, file_size, file_directory) in files:
        fp.add_file_to_directory(file_name, file_size, file_directory)

    print(fp.all_files)
    print("\n")

    print("Total Processed -: \n\t", fp.get_total_file_system_size())
    print(f"Top-{top_k} collections are -: \n\t ", fp.get_top_k_directory(top_k=top_k))
