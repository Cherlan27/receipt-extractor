from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

CLASSIFY_SYSTEM_PROMPT = dedent("""\
    Du bekommst einen OCR geparsten Einkaufszettel vom User. Deine Aufgabe ist es,
    die unterschiedlichen eingekauften Produkte in Kategorien einzuteilen.
    Diese Kategorien sind: Süßigkeiten, Obst, Gemüse, Fleisch, Brot, Anderes.

    Gebe ein gültiges JSON-Objekt mit einer geordneten Liste "items" dafür aus,
    ohne weiteren Text. Behalte die Reihenfolge des Einkaufszettels bei.
    Der Preis ist eine Zahl mit Punkt als Dezimaltrenner.

    Beispiel:
    Eingabe:
    Apfel 1.99
    Milch 1.29
    Gummibärchen 0.99

    Ausgabe:
    {{"items": [
        {{"product": "Apfel", "price": 1.99, "category": "Obst"}},
        {{"product": "Milch", "price": 1.29, "category": "Anderes"}},
        {{"product": "Gummibärchen", "price": 0.99, "category": "Süßigkeiten"}}
    ]}}

    {format_instructions}

    Der Einkaufszettel wird dir vom User geschickt.
    """)

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFY_SYSTEM_PROMPT),
        ("human", "{receipt_text}"),
    ]
)
