"""Script de debug pour tester la connexion Ollama."""

import ollama
import json

print("=== Debug Ollama Connection ===\n")

try:
    print("1. Test ollama.list()...")
    response = ollama.list()

    print(f"   Type de réponse : {type(response)}")
    print(f"   Contenu brut : {response}")
    print()

    # Si c'est un dict
    if isinstance(response, dict):
        print("   Structure dict détectée")
        print(f"   Clés disponibles : {list(response.keys())}")
        models = response.get('models', [])
        print(f"   Nombre de modèles : {len(models)}")

        if models:
            print("\n   Modèles trouvés :")
            for model in models:
                print(f"     - Type: {type(model)}")
                print(f"       Contenu: {model}")
        else:
            print("   ⚠️  Liste de modèles vide")

    # Si c'est une liste
    elif isinstance(response, list):
        print("   Structure list détectée")
        print(f"   Nombre d'éléments : {len(response)}")

        if response:
            print("\n   Éléments :")
            for item in response:
                print(f"     - Type: {type(item)}")
                print(f"       Contenu: {item}")
        else:
            print("   ⚠️  Liste vide")

    else:
        print(f"   ⚠️  Type inattendu : {type(response)}")

    print("\n2. Test direct avec ollama.show()...")
    try:
        info = ollama.show("llava:7b")
        print(f"   ✓ llava:7b existe !")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   ✗ llava:7b non accessible : {e}")

except Exception as e:
    print(f"\n❌ Erreur : {e}")
    print(f"Type: {type(e)}")
    import traceback
    traceback.print_exc()

print("\n=== Fin du debug ===")
