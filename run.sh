#!/bin/bash

echo "📦 Création d'un environnement virtuel..."
python3 -m venv venv

echo "🔁 Activation de l'environnement..."
source venv/bin/activate

echo "⬇️ Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Lancement de l'application Flask..."
python app.py
