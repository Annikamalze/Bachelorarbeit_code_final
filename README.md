# Integration von PaccMann in DrEval - Bachelorarbeit
Dieses Repository enthält den Code für meine Bachelorarbeit zur Implementierung und Evaluierung des PaccMann-Modells in die DrEval-Pipeline. 

Die DrEval-Pipeline ist hier zu finden: **[DrEval GitHub](https://github.com/daisybio/drevalpy)**.
Das PaccMann-Modell ist in DrEval unter "models/pacc_mann" zu finden. Der Quellcode des implementierten Modells befindet sich in der PaccMann.py und nutzt die, sich im Ordner "hilfsfunktionen" befindenden Dateien des ursprünglichen **[PaccMann GitHubs](https://github.com/drugilsberg/paccmann)**. 

Im Ordner "ResponseData" sind alle Dateien zur Erstellung der verschiedenen Datensatzgrößen zur Evaluierung des Modells abgelegt. 

Zur Vorbereitung der SMILES-Sequenzen wurde die "paccmann_smiles_embedding.py" genutzt. 
Alle verwendeten Daten sind unter "data" des DrEval-Moduls einzusehen.
