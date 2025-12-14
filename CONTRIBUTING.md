# Guide de contribution

Merci de votre intérêt pour contribuer à EIMLIA-TEU !

## Comment contribuer

### Signaler un bug

1. Vérifiez que le bug n'a pas déjà été signalé dans les [Issues](https://github.com/chu-lille/eimlia-teu/issues)
2. Créez une nouvelle issue avec:
   - Une description claire du problème
   - Les étapes pour reproduire
   - Le comportement attendu vs observé
   - Votre environnement (OS, Python, versions)

### Proposer une amélioration

1. Ouvrez une issue pour discuter de l'idée
2. Attendez la validation avant de coder
3. Suivez les conventions du projet

### Soumettre du code

1. **Fork** le repository
2. Créez une branche: `git checkout -b feature/ma-fonctionnalite`
3. Commitez: `git commit -m "feat: description"`
4. Push: `git push origin feature/ma-fonctionnalite`
5. Ouvrez une **Pull Request**

## Standards de code

### Style

- Python 3.10+
- Formatage: **Black** (ligne max 100)
- Imports: **isort** (profile black)
- Linting: **flake8**
- Types: **mypy**

```bash
# Installer les outils
pip install -e ".[dev]"
pre-commit install

# Vérifier le code
make lint
make format
```

### Conventions de commit

Nous suivons [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation
- `style:` formatage
- `refactor:` refactoring
- `test:` tests
- `chore:` maintenance

### Tests

- Tous les tests doivent passer: `make test`
- Couverture minimum: 80%
- Nouveaux fichiers = nouveaux tests

```bash
# Lancer les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

### Documentation

- Docstrings Google style
- README à jour
- Exemples fonctionnels

```python
def ma_fonction(param: str) -> int:
    """
    Description courte.
    
    Description longue si nécessaire.
    
    Args:
        param: Description du paramètre
        
    Returns:
        Description du retour
        
    Raises:
        ValueError: Si param est vide
        
    Example:
        >>> ma_fonction("test")
        42
    """
    pass
```

## Structure du projet

```
eimlia-teu/
├── src/
│   ├── models/        # Modèles IA
│   ├── simulation/    # Simulation hybride
│   ├── process_mining/ # PM4Py
│   ├── api/           # FastAPI
│   └── utils/         # Utilitaires
├── tests/             # Tests
├── scripts/           # Scripts CLI
├── docs/              # Documentation
└── docker/            # Docker
```

## Processus de review

1. Le code est revu par au moins 1 mainteneur
2. Les tests doivent passer (CI)
3. Le code doit être formaté
4. La documentation doit être à jour

## Questions?

- Ouvrez une [Discussion](https://github.com/chu-lille/eimlia-teu/discussions)
- Contactez: edouard1.lansiaux@chu-lille.fr

## Licence

En contribuant, vous acceptez que votre code soit sous licence MIT.
