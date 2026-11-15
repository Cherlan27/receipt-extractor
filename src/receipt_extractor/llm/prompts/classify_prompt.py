from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

CLASSIFY_SYSTEM_PROMPT = dedent("""\
    Du bekommst einen OCR geparsten Einkaufszettel vom User. Deine Aufgabe ist es,
    die Eckdaten des Einkaufs zu lesen und die unterschiedlichen eingekauften
    Produkte in Kategorien einzuteilen.
    Diese Kategorien sind: Süßigkeiten, Obst, Gemüse, Fleisch, Brot, Anderes.

    Gebe ein gültiges JSON-Objekt mit den Eckdaten und einer geordneten Liste
    "items" dafür aus, ohne weiteren Text. Behalte die Reihenfolge des
    Einkaufszettels bei.
    - "purchased_at": Kaufdatum als "YYYY-MM-DD", mit Uhrzeit als
      "YYYY-MM-DDTHH:MM:SS", falls der Beleg sie zeigt.
    - "store": Name des Geschäfts, wie auf dem Beleg gedruckt.
    - "total": der gezahlte Gesamtbetrag (z.B. nach ZU ZAHLEN).
    - Steht ein Wert nicht auf dem Beleg, ist er null. Erfinde nichts.
    - "quantity": ganze Zahl, Standard 1. Eine Zeile wie "2 x 1.39" davor
      bedeutet Menge 2.
    - "price": Gesamtpreis der Zeile (Menge mal Einzelpreis), also der Betrag
      rechts in der Zeile.
    - Die Ziffer hinter dem Euro-Zeichen (z.B. "1.59 € 2") ist die Steuerklasse,
      keine Menge.
    Alle Beträge sind Zahlen mit Punkt als Dezimaltrenner.

    Beispiel:
    Eingabe:
    ALDI
    Datum 25.02.26 19:33 Uhr
    Apfel 1.99
    2 x 0.65
    Milch 1.30
    Gummibärchen 0.99
    ZU ZAHLEN 4.28

    Ausgabe:
    {{"purchased_at": "2026-02-25T19:33:00", "store": "ALDI", "total": 4.28, "items": [
        {{"product": "Apfel", "quantity": 1, "price": 1.99, "category": "Obst"}},
        {{"product": "Milch", "quantity": 2, "price": 1.30, "category": "Anderes"}},
        {{"product": "Gummibärchen", "quantity": 1, "price": 0.99, "category": "Süßigkeiten"}}
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
