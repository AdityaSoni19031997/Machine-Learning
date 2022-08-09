from collections import defaultdict
from dataclasses import dataclass


class EnglishSpanishTranslator:
    # translate between two languages only (warm-up)
    def __init__(self) -> None:
        self.spanish_to_english = defaultdict(str)
        self.english_to_spanish = defaultdict(str)

    # adds a bi-directional mapping between these words to the knowledge base
    def addTranslation(self, englishWord: str, spanishWord: str) -> None:
        self.english_to_spanish[englishWord] = spanishWord
        self.spanish_to_english[spanishWord] = englishWord

    # gets english for a spanish word else raise an exception
    def getEnglishWord(self, spanishWord: str):
        if spanishWord in self.spanish_to_english:
            return self.spanish_to_english.get(spanishWord)
        else:
            # spanmish word is not there, so we don't have the english translation for it.
            raise KeyError(
                f"We don't have the translation for this spanish_word -: {spanishWord} to english language unfortunately"
            )

    # gets spanish for a english word else raise an exception
    def getSpanishWord(self, englishWord: str):
        if englishWord in self.english_to_spanish:
            return self.english_to_spanish.get(englishWord)
        else:
            # English word is not there, so we don't have the Spanish translation for it.
            raise KeyError(
                f"We don't have the translation for this english_word -: {englishWord} to Spanish language unfortunately"
            )


def test_eng_spa_translator():
    translator = EnglishSpanishTranslator()

    translator.addTranslation("eng_word_1", "sp_word_1")
    translator.addTranslation("eng_word_2", "sp_word_2")
    translator.addTranslation("eng_word_4", "sp_word_4")

    translator.getEnglishWord("sp_word_1")
    translator.getEnglishWord("sp_word_2")

    translator.getSpanishWord("eng_word_2")
    translator.getSpanishWord("eng_word_4")

    translator.getSpanishWord("eng_word")  # error
 

# A generic idea to implement the same acorss any number of language...

@dataclass
class Node:

    language: str
    word: str

    def __hash__(self) -> int:
        # dummy hashing...
        # not needed if you use unsafe_hash = True in dataclass decorator.
        return hash(self.language + self.word)


class GenericTranslator:
    def __init__(self) -> None:
        self.langage_map = defaultdict(list)

    def add(self, from_lang: str, from_word: str, to_lang: str, to_word: str) -> None:
        # add the translation from one language to another to the graph
        self._add(Node(from_lang, from_word), Node(to_lang, to_word))

    def _add(self, from_node: Node, to_node: Node) -> None:
        # bi-directional mapping added to the grpah
        self.langage_map[from_node].append(to_node)
        self.langage_map[to_node].append(from_node)

    def translate(self, from_lang: str, to_lang: str, word: str):
        # translate from one language to another if we are expected to find that conversion
        visited = set()
        result = self.dfs(Node(from_lang, word), to_lang, visited)
        if result == None:
            return f"Sorry cannot translate {word} to {to_lang} language...."
        return f"{word} in {from_lang}, when succesfully translated to {to_lang} becomes {result} !!!"

    def dfs(self, node: Node, to_lang: str, seen: set) -> str:
        if node == None:
            return None
        if node.language == to_lang:
            return node.word
        if self.langage_map.get(node) == None:
            return None
        if node in seen:
            return None

        # mark as visited
        seen.add(node)

        # scan the graph
        for neighbour_node in self.langage_map.get(node):
            result = self.dfs(neighbour_node, to_lang, seen)
            if result != None:
                return result

        # backtrack
        seen.remove(node)
        return None


def main():
    # simple helper method to dry run...
    translator = GenericTranslator()
    translator.add("English", "Hello", "Spanish", "Hola")
    translator.add("Spanish", "Hola", "Hindi", "Namaste")

    print(translator.translate("Hindi", "English", "Namaste"))
    print(translator.translate("Hindi", "Spanish", "Namaste"))
    print(translator.translate("English", "Hindi", "Hello"))
    print(translator.translate("English", "Spanish", "Hello"))
    print(translator.translate("Spanish", "English", "Hola"))
    print(translator.translate("Spanish", "Hindi", "Hola"))
    print(translator.translate("Spanish", "French", "Hola"))
    print(translator.translate("Spanish", "Hindi", "Holaa"))


main()
