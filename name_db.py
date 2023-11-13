import random
from collections import Counter
import csv
from itertools import groupby
import bisect

class NameDB():
    def __init__(self, path):
        # This file: https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv
        self.data = [row for row in csv.DictReader(open(path))]
        grouped = groupby(self.data, key=lambda row: row["name"])
        self.data = []
        for key, to_reduce in grouped:
            to_reduce = list(to_reduce)
            sex = Counter(row["sex"] for row in to_reduce).most_common(1)[0][0]
            self.data.append({"sex": sex, "name": key, "percent": sum(float(row["percent"]) for row in to_reduce)}) 
        self.randomizer = random.Random()
        self.probs = [row["percent"] for row in self.data]
        self.names = [row["name"] for row in self.data]
        self.name_to_sex = {row["name"].lower(): row["sex"] for row in self.data}
        self.male_names = [row["name"] for row in self.data if row["sex"] == "boy"]
        self.female_names = [row["name"] for row in self.data if row["sex"] == "girl"]
        self.male_probs = [float(row["percent"]) for row in self.data if row["sex"] == "boy"]
        self.female_probs = [float(row["percent"]) for row in self.data if row["sex"] == "girl"]
        self.probs = self._to_cumulative(self.probs)
        self.male_probs = self._to_cumulative(self.male_probs)
        self.female_probs = self._to_cumulative(self.female_probs)
        self.max_probs = max(self.probs)
        self.max_male_probs = max(self.male_probs)
        self.max_female_probs = max(self.female_probs)

    def _to_cumulative(self, list_):
        total = 0
        out = []
        for e in list_:
            total += e
            out.append(total)
        return out

    def random_choice(self, list_, probs, max_val):
        rand = self.randomizer.random() * max_val
        i = bisect.bisect_left(probs, rand)
        return list_[i]

    def random_name_with_sex(self, sex):
        if sex is not None:
            names = self.male_names if sex == "boy" else self.female_names
            probs, max_val = (self.male_probs, self.max_male_probs) if sex == "boy" else (self.female_probs, self.max_female_probs)
            return self.random_choice(names, probs, max_val)
        else:
            return self.random_name()

    def random_name(self):
        choice = self.random_choice(self.names, self.probs, self.max_probs)
        return choice

    def get_sex_for_name(self, name):
        return self.name_to_sex.get(name.lower().strip())