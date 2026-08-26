# Preprocessors

In **KGATE**, a preprocessor is a module that will generate initial features for your graph *before* the beginning of the training. These features may already be embeddings, but can be any kind of numerical representation. This step is completely optionnal, but using a random-walk based preprocessor has been shown to **speed up** the training of a model by providing it with informed initial embeddings instead of purely random ones. If you are looking for modules to encode the initial features into learned embeddings, see the **[encoders](./encoders.md)** page.

```{currentmodule} kgate.preprocessors
```

```{autosummary}
   :nosignatures:
   :toctree: ../generated

   Node2Vec
```

