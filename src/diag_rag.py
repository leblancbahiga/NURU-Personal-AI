#!/usr/bin/env python3
"""
diag_rag.py — Diagnostic complet du pipeline RAG de NURU.
Vérifie que les documents indexés sont bien retrouvés et transmis au modèle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag import VectorStore, get_embedder

def main():
    print("=" * 60)
    print("  DIAGNOSTIC RAG — NURU")
    print("=" * 60)
    
    # 1. Vérifier que ChromaDB est accessible
    try:
        store = VectorStore()
        print(f"\n✅ ChromaDB connecté : {store.persist_dir}")
    except Exception as e:
        print(f"\n❌ Impossible de se connecter à ChromaDB : {e}")
        return
    
    # 2. Compter les documents
    n_docs = store.count_documents()
    n_corr = store.count_corrections()
    n_conv = store.col_conversations.count()
    print(f"\n📊 Statistiques de la base :")
    print(f"   - Documents indexés : {n_docs} chunks")
    print(f"   - Corrections       : {n_corr}")
    print(f"   - Conversations     : {n_conv}")
    
    if n_docs == 0:
        print("\n❌ PROBLÈME : Aucun document indexé dans ChromaDB !")
        print("   → Lance d'abord : python3 src/ingestion.py")
        return
    
    # 3. Afficher les fichiers indexés
    print(f"\n📁 Fichiers indexés (échantillon) :")
    try:
        sample = store.col_documents.peek(limit=10)
        filenames = set()
        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                fn = meta.get("filename", "?")
                filenames.add(fn)
        for fn in sorted(filenames):
            print(f"   • {fn}")
    except Exception as e:
        print(f"   ⚠ Erreur peek : {e}")
    
    # 4. Test de recherche avec différents seuils
    test_query = input("\n🔍 Entre une question test (ou Entrée pour 'Qui est Leblanc ?') : ").strip()
    if not test_query:
        test_query = "Qui est Leblanc ?"
    
    print(f"\n🔍 Recherche pour : '{test_query}'")
    print("-" * 40)
    
    # Recherche SANS seuil (threshold=0) pour voir TOUS les résultats
    results_raw = store.search(test_query, k=10, threshold=0.0)
    
    if not results_raw:
        print("❌ AUCUN résultat retourné par ChromaDB (même sans seuil) !")
        print("   → Le problème est dans l'embedding ou ChromaDB.")
        return
    
    print(f"\n📋 {len(results_raw)} résultats trouvés (sans seuil) :\n")
    for i, r in enumerate(results_raw):
        score = r["score"]
        comp = r.get("hybrid_components", {})
        src = r["metadata"].get("filename", "?")
        text_preview = r["text"][:120].replace("\n", " ")
        
        # Marquer visuellement les résultats au-dessus/en-dessous des seuils
        if score >= 0.50:
            marker = "🟢"  # Passerait le seuil 0.50
        elif score >= 0.40:
            marker = "🟡"  # Passerait 0.40 mais pas 0.50
        elif score >= 0.30:
            marker = "🟠"  # Passerait 0.30 mais pas 0.40
        else:
            marker = "🔴"  # Filtré même avec 0.30

        print(f"  {marker} #{i+1} — Score: {score:.4f}  [emb:{comp.get('embedding',0):.3f}  kw:{comp.get('keyword',0):.3f}  rec:{comp.get('recency',0):.3f}]")
        print(f"     Source: {src}")
        print(f"     Texte : {text_preview}...")
        print()
    
    # 5. Résumé et recommandation
    above_050 = sum(1 for r in results_raw if r["score"] >= 0.50)
    above_040 = sum(1 for r in results_raw if r["score"] >= 0.40)
    above_030 = sum(1 for r in results_raw if r["score"] >= 0.30)
    
    print("=" * 60)
    print(f"  RÉSUMÉ DES SEUILS :")
    print(f"  🟢 Score ≥ 0.50 : {above_050} résultats")
    print(f"  🟡 Score ≥ 0.40 : {above_040} résultats")
    print(f"  🟠 Score ≥ 0.30 : {above_030} résultats")
    print("=" * 60)
    
    if above_050 == 0 and above_040 > 0:
        print("\n⚠ DIAGNOSTIC : Le seuil 0.50 est TROP ÉLEVÉ pour tes documents.")
        print("   → Recommandation : baisser le seuil à 0.40")
    elif above_040 == 0 and above_030 > 0:
        print("\n⚠ DIAGNOSTIC : Le seuil 0.40 est TROP ÉLEVÉ pour tes documents.")
        print("   → Recommandation : garder le seuil à 0.30")
    elif above_050 > 0:
        print("\n✅ Les documents sont bien trouvés avec le seuil 0.50.")
        print("   Si le modèle hallucine encore, le problème est dans le prompting.")
    else:
        print("\n❌ Aucun résultat pertinent même à 0.30. Vérifie l'indexation.")
    
    # 6. Test du format de contexte
    if results_raw:
        # Simuler ce que le modèle recevrait
        filtered = [r for r in results_raw if r["score"] >= 0.40]
        if filtered:
            ctx = store.format_context(filtered[:5])
            print(f"\n📝 Contexte qui serait injecté dans le prompt ({len(ctx)} caractères) :")
            print("-" * 40)
            print(ctx[:500])
            if len(ctx) > 500:
                print(f"... [{len(ctx) - 500} caractères supplémentaires]")

if __name__ == "__main__":
    main()
