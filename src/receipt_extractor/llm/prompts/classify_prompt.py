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
    - "quantity": Standard ist 1. Nur wenn direkt über dem Produktnamen eine Zeile
      "N x Einzelpreis" steht, ist die Menge N.
    - "price": Gesamtpreis der Zeile, immer der Betrag unter dem Produktnamen.
    - Die Ziffer hinter dem Betrag mit Euro-Zeichen ist die Steuerklasse, nie die
      Menge. Beispiele:
      "Cola" / "1.59 € 2" → quantity 1, price 1.59
      "3 x 0.40 €" / "Brötchen" / "1.20 € 1" → quantity 3, price 1.20
    Alle Beträge sind Zahlen mit Punkt als Dezimaltrenner.

    Beispiel:
    Eingabe:
    ALDI
    Apfel
    1.99 € 1
    Cola Zero
    1.59 € 2
    2 x 0.65 €
    Milch
    1.30 € 1
    ZU ZAHLEN
    4.88 €
    Datum 25.02.26
    19:33 Uhr

    Ausgabe:
    {{"purchased_at": "2026-02-25T19:33:00", "store": "ALDI", "total": 4.88, "items": [
        {{"product": "Apfel", "quantity": 1, "price": 1.99, "category": "Obst"}},
        {{"product": "Cola Zero", "quantity": 1, "price": 1.59, "category": "Anderes"}},
        {{"product": "Milch", "quantity": 2, "price": 1.30, "category": "Anderes"}}
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
