# Architect

The **Architect** class is the centerpiece when using **KGATE** as a framework. It holds everything form the knowledge graph to the model weights and the configuration. For a simple utilisation, only the function `train_model`, `test` et `infer` are relevant, as well as `get_embeddings` to obtain the final latent space.

```{autoclass} kgate.architect.Architect
:members:
```