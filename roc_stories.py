import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

@dataclass
class Story():
    story_id: str
    sentences: list[str]
    candidate_endings: list[str]
    label: Optional[int]  # index into candidate endings

    @classmethod
    def from_json(cls, data):
        return cls(
            story_id=data["InputStoryid"],
            sentences=[data[f"InputSentence{n}"] for n in range(1, 5)],
            candidate_endings=[data[f"RandomFifthSentenceQuiz{n}"] for n in range(1, 3)],
            label=int(data["AnswerRightEnding"] or 0) - 1
        )

class ROCStoriesDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "dev") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        split_file_name = {
            "test": "cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv",
            "dev": "cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
            "train": "ROCStories_winter2017 - ROCStories_winter2017.csv",
        }[split]
        columns = [
            "InputStoryid",
            "InputSentence1",
            "InputSentence2",
            "InputSentence3",
            "InputSentence4",
            "RandomFifthSentenceQuiz1",
            "RandomFifthSentenceQuiz2",
            "AnswerRightEnding",
        ]
        self.file_path = self.data_dir / split_file_name
        self.reader = csv.DictReader(open(self.file_path), fieldnames=columns)
        self.data = []
        next(self.reader)
        for line in self.reader:
            self.data.append(Story.from_json(line))
        super().__init__()

    def __getitem__(self, index):
        return self.data[index]
    

if __name__ == "__main__":
    ds = ROCStoriesDataset("../roc_stories", "dev")
    for item in ds:
        anchor = " ".join(item.sentences)
        choices = [anchor + " " + s for s in item.candidate_endings]