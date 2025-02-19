import numpy as np
import pandas as pd
import os
import torch
import yaml
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Optional
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_reduce_gene_features
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
import sys
sys.path.append("/storage/mi/malzea03/Bachelorarbeit_code/drevalpy/models/pacc_mann/hilfsfunktionen")
from paccmann_utils import PaccMannV2

## Chat GPT wurde zum Debuggen dieser Datei benutzt ##

class PaccMann(DRPModel):
    """
    Die Klasse für das PaccMann Modell zur Vorhersage der Medikamentenreaktion.
    """
    cell_line_views = ["gene_expression"]
    drug_views = ["SMILES"]
    is_single_drug_model=False

    def __init__(self):
        """
        Initialisiert das Modell.
        """
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__()
        self.model = None  # -> Modell wird in der build_model-Methode aufgebaut
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.is_single_drug_model=False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.
        :return: model name
        """
        return "PaccMann"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Baut das Modell mit den angegebenen Hyperparametern auf.

        :param hyperparameters: Enthält die Hyperparameter für das Modell, z.B. Anzahl der Schichten, Dropout-Rate, etc.
        """
        self.hyperparameters = hyperparameters

        layers = []
        input_size = 2089+231 

        self.model = PaccMannV2(hyperparameters)
        self.epochs = hyperparameters['epochs']
        self.batch_size = 32  
        self.lr = hyperparameters['learning_rate']

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Lädt Zelllinienmerkmale vor dem Training oder der Vorhersage.

        :param data_path: Pfad zu den Daten, z.B. "data/"
        :param dataset_name: Name des Datasets, z.B. "GDSC2"
        :returns: FeatureDataset mit Zelllinienmerkmalen
        """
        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list="2128_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_smiles_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Lädt die SMILES-Merkmale für das angegebene Dataset.

        :param data_path: Pfad zu den Daten, z.B. "data/"
        :param dataset_name: Name des Datasets, z.B. "GDSC2"
        :returns: FeatureDataset mit den Medikamenten-SMILES
        """
        if dataset_name == "GDSC2":
            smiles_file = "processed_smiles_gdsc2.csv"
        elif dataset_name == "CCLE":
            smiles_file = "processed_smiles_ccle.csv"
        else:
            raise ValueError(f"Unbekanntes Dataset: {dataset_name}")

        smiles = pd.read_csv(os.path.join(data_path, dataset_name, smiles_file), index_col=0)

        return FeatureDataset(
            features={drug: {"SMILES": smiles.loc[drug].values} for drug in smiles.index}
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Lädt die Wirkstoffmerkmale vor dem Training oder der Vorhersage.

        :param data_path: Pfad zu den Daten, z.B. "data/"
        :param dataset_name: Name des Datasets, z.B. "GDSC2"
        :returns: FeatureDataset oder None
        """
        return self.load_drug_smiles_features(data_path, dataset_name)

    def train(
            self, 
            output: DrugResponseDataset, 
            cell_line_input: FeatureDataset, 
            drug_input: FeatureDataset | None = None, 
            output_earlystopping: DrugResponseDataset | None = None
            ) -> None:
        """
        Trainiert das Modell.

        :param output: Die Trainingsdaten für die Antwort.
        :param cell_line_input: Die Zelllinien-Eingabedaten.
        :param drug_input: Die Wirkstoff-Eingabedaten.
        """

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="SMILES",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        gene_expression = x[:, :2089]  
        smiles = x[:, 2089:]       
        if gene_expression.shape[0] != smiles.shape[0] or gene_expression.shape[0] != len(output.response):
            raise ValueError("Die Dimensionen von Gene Expression, SMILES und Response stimmen nicht überein.")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(gene_expression, dtype=torch.float32).to(self.DEVICE),
                torch.tensor(smiles, dtype=torch.float32).to(self.DEVICE), 
                torch.tensor(output.response, dtype=torch.float32).to(self.DEVICE)
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(gene_expression, dtype=torch.float32).to(self.DEVICE),
                torch.tensor(smiles, dtype=torch.float32).to(self.DEVICE), 
                torch.tensor(output.response, dtype=torch.float32).to(self.DEVICE)
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()  

        best_val_loss = float('inf')  
        epochs_without_improvement = 0  
        early_stopping_patience = 5  

        for epoch in range(self.epochs):
            epoch_loss = 0
            self.model.train()  
            for gene_expression_batch, smiles_batch, response_batch in train_loader:
                optimizer.zero_grad()

                predicted_response, _ = self.model(smiles_batch, gene_expression_batch)

                loss = criterion(predicted_response.view(-1), response_batch)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss}")

            # Validierung 
            self.model.eval()  
            with torch.no_grad():  
                valid_loss = 0
                for gene_expression_batch, smiles_batch, response_batch in valid_loader:
                    predicted_response, _ = self.model(smiles_batch, gene_expression_batch)
                    loss = criterion(predicted_response.view(-1), response_batch)
                    valid_loss += loss.item()

                avg_valid_loss = valid_loss / len(valid_loader)
                print(f"Validation Loss: {avg_valid_loss}")

            # Early Stopping Überprüfung
            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping nach {epoch + 1} Epochen ohne Verbesserung.")
                    break  

               

    def predict(
        self, 
        cell_line_ids: np.ndarray, 
        drug_ids: np.ndarray, 
        cell_line_input: FeatureDataset, 
        drug_input: FeatureDataset | None = None
        ) -> np.ndarray:
        """
        Sagt die Antwort für die gegebenen Eingaben voraus.

        :param cell_line_ids: Liste der Zelllinien-IDs.
        :param drug_ids: Liste der Wirkstoff-IDs.
        :param cell_line_input: Eingabedaten für die Zelllinien.
        :param drug_input: Eingabedaten für die Wirkstoffe.
        :returns: Vorhergesagte Antwort
        """
        self.model.eval()

        if drug_input is None:
            raise ValueError("drug_input (SMILES) sind erforderlich, um Vorhersagen zu machen.")

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="SMILES",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        gene_expression = x[:, :2089]  
        smiles = x[:, 2089:]

        with torch.no_grad():  # Deaktiviert das Gradienten-Tracking
            gene_expression_tensor = torch.tensor(gene_expression, dtype=torch.float32).to(self.DEVICE)  
            smiles_tensor = torch.tensor(smiles, dtype=torch.float32).to(self.DEVICE)
            
            predictions,_ = self.model(smiles_tensor, gene_expression_tensor)
            predictions = predictions.detach().cpu().numpy()           

        return predictions


    def load_hyperparameters(file_path: str) -> dict:
        """
        Lädt die Hyperparameter aus einer YAML-Datei.

        :param file_path: Pfad zur Hyperparameter-Datei.
        :returns: Hyperparameter als Dictionary
        """
        with open(file_path, 'r') as file:
            hyperparameters = yaml.safe_load(file)
        return hyperparameters
