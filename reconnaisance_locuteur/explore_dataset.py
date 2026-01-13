import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import de la configuration
from config import DATASET_ROOT, PROJECT_ROOT

print("🔍 EXPLORATION DATASET VOXCELEB")

# Vérifier le chemin et explorer la structure
if DATASET_ROOT.exists():
    print(f" Dataset trouvé: {DATASET_ROOT}\n")
    
    # Lister tous les locuteurs (dossiers commençant par 'id')
    all_speakers = sorted([d.name for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name.startswith('id')])
    
    print(f"Nombre total de locuteurs: {len(all_speakers)}")
    print(f" Premiers locuteurs: {all_speakers[:5]}\n")
    
    if len(all_speakers) > 0:
        # Explorer un locuteur exemple
        sample_speaker = all_speakers[0]
        sample_path = DATASET_ROOT / sample_speaker
        
        # Lister les vidéos
        videos = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])
        
        print(f" Locuteur exemple: {sample_speaker}")
        print(f" Nombre de vidéos: {len(videos)}")
        print(f" Exemples: {videos[:3]}\n")
        
        if len(videos) > 0:
            # Compter fichiers audio dans une vidéo
            sample_video = sample_path / videos[0]
            wav_files = list(sample_video.glob('*.wav'))
            
            print(f" Fichiers audio dans {videos[0]}: {len(wav_files)}")
            
            if wav_files:
                # Analyser un fichier audio exemple
                example_wav = wav_files[0]
                print(f"\n Analyse d'un fichier exemple: {example_wav.name}")
                
                try:
                    y, sr = librosa.load(str(example_wav), sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    print(f"   ├─ Sample rate: {sr} Hz")
                    print(f"   ├─ Durée: {duration:.2f}s")
                    print(f"   ├─ Samples: {len(y)}")
                    print(f"   └─ Chemin: {example_wav.parent.name}/{example_wav.name}")
                    
                except Exception as e:
                    print(f" Erreur lecture audio: {e}")
            
            # Statistiques complètes

            print(" STATISTIQUES DATASET")

            total_audio = 0
            speaker_stats = []
            
            print("Analyse en cours...")
            for speaker in tqdm(all_speakers[:10], desc="Échantillonnage"):  # Analyse 10 premiers
                speaker_path = DATASET_ROOT / speaker
                audio_count = len(list(speaker_path.glob('**/*.wav')))
                total_audio += audio_count
                speaker_stats.append({'speaker': speaker, 'audio_files': audio_count})
            
            df_stats = pd.DataFrame(speaker_stats)
            
            print(f"\n Statistiques (10 premiers locuteurs):")
            print(f"   ├─ Total fichiers audio: {total_audio}")
            print(f"   ├─ Moyenne par locuteur: {df_stats['audio_files'].mean():.0f}")
            print(f"   ├─ Min: {df_stats['audio_files'].min()}")
            print(f"   └─ Max: {df_stats['audio_files'].max()}")
            
        else:
            print(f" Aucune vidéo trouvée pour {sample_speaker}")
    else:
        print("Aucun locuteur trouvé dans le dataset")

    print(" STRUCTURE ATTENDUE:")

    print("wav/")
    print("├── id10001/")
    print("│   ├── video1/")
    print("│   │   ├── 00001.wav")
    print("│   │   └── 00002.wav")
    print("│   └── video2/")
    print("└── id10002/")

    
else:
    print(f" Dataset NON trouvé à: {DATASET_ROOT}\n")
    print(" Solutions possibles:")
    print("   1. Vérifie que le dataset VoxCeleb est bien téléchargé")
    print("   2. Place le dataset dans le bon dossier")
    print(f"   3. Chemin attendu: {DATASET_ROOT}\n")
    
    # Explorer le dossier parent
    print(f"📂 Contenu du dossier root ({PROJECT_ROOT}):")
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir():
            print(f"    {item.name}/")
        else:
            print(f"    {item.name}")
    
    print("\n Structure recommandée:")
    print("Traitement audio/")
    print("├── wav/  ← Place tes fichiers VoxCeleb ici")
    print("├── output/")
    print("└── reconnaisance_locuteur/")
    print("    └── dataset_kaggle_VoxCeleb.py")

print("\n Exploration terminée\n")
