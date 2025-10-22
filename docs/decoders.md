# Decoders

While it is possible to implement your own, KGATE includes several decoders implemented out of the box.

## Translational

```{currentmodule} kgate.decoders.translational
```

```{autosummary}
   :nosignatures:
   :toctree: ../generated

   TransE
   TransH
   TransR
   TransD
   TorusE
```

## Bilinear

```{currentmodule} kgate.decoders.bilinear
```

```{autosummary}
   :nosignatures:
   :toctree: ../generated

   RESCAL
   DistMult
   ComplEx
```

## Convolutional

```{currentmodule} kgate.decoders.convolutional
```

```{autosummary}
   :nosignatures:
   :toctree: ../generated

   ConvKB
```

For a complete signature of the decoders, see []

## Build your own decoder

All KGATE decoders are built from [TorchKGE's decoders](https://github.com/torchkge-team/torchkge/tree/master/torchkge/models) and thus use the same methods, with slight modifications. It is recommended, but not mandatory, that decoders inherit from at least TorchKGE's [Model](https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/interfaces.py#L13), and when applicable the relevant [TranslationModel](https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/interfaces.py#L177) or [BilinearModel](https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/interfaces.py#L275). In any case, a decoder class must have the following methods:

- `score`: Method called by the Architect's {#kgate.architect.Architect.scoring_function} method. Given the current batch embeddings and indices, it will output the score computed by the decoder as a tensor.
- `get_embeddings`: While the main embeddings are stored in the {#kgate.architect.Architect} object, some decoders use additionnal embeddings that are not encoded by GNN beforehands. Decoders with no additionnal embeddings should return `None`.
- `normalize_params`: a function taking the related embedding tensors as argument. When used from within the Architect, the entity and relation embeddings will be passed to this function, which normalizes them in addition to potential decoder-specific embeddings, before returning both. If the decoder requires no normalization, this method can be omitted or return the embeddings without alteration.
- `inference_prepare_candidates`: Helper function for the link prediction evaluation. Given the current batch indices and full embeddings, returns the batch embedding as well as all possible candidates to evaluate link prediction against.