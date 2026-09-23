JUVENS AI BACKEND FINAL
1. Remplace main.py sur Render.
2. Vérifie requirements.txt.
3. Variables Render:
   GEMINI_API_KEY = ta clé
   GEMINI_MODEL = gemini-3-flash
   JWT_SECRET = une longue valeur secrète
   ADMIN_EMAIL = ton email admin
4. Le POST /chat est maintenant utilisable sans connexion.
5. Si un JWT existe, l'historique peut être sauvegardé.
6. Les images et textes de fichiers joints sont transmis au modèle.
7. La clé Gemini reste uniquement sur Render.
