# Decoders

While it is possible to implement your own, KGATE includes several decoders implemented out of the box. When using a deep learning [encoder](./encoders.md), the most basic decoder to use is [DistMult](#bilinear). Either way, it is recommended to benchmark multiple decoder to find out which one learns the best.

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
- `normalize_params`: a method taking the related embedding tensors as argument. When used from within the Architect, the entity and relation embeddings will be passed to this method, which normalizes them in addition to potential decoder-specific embeddings, before returning both. If the decoder requires no normalization, this method can be omitted or return the embeddings without alteration.
- `inference_prepare_candidates`: Helper method for the link prediction evaluation. Given the current batch indices and full embeddings, returns the batch embedding as well as all possible candidates to evaluate link prediction against.
- `inference_scoring_function`: Computes the score of a given triplet, using as input the output of `inference_prepare_candidates`. This method is already implemented in **TorchKGE** and doesn't need to be reimplemented in most cases.

A decoder *must not* contain the entity or relation main embeddings, but may use additionnal embeddings. However, if these additionnal embeddings are meant to be encoded, for example in the case of an imaginary embedding space, you should add the `embedding_spaces` property to the decoder. Then, the number of embedding dimension in the output of the encoder will correspond to `emb_dim * decoder.embedding_spaces` and split accordingly in the different spaces.

```{hint} Example
**ComplEx** uses a real embedding space and an imaginary one. Thus, the definition of the class is like so:

   ```python
      class ComplEx(ComplExModel):
         def __init__(self, embedding_dimensions: int):
              super().__init__()
              self.embedding_dimensions = embedding_dimensions
              self.embedding_spaces = 2
   ```

If the embedding dimension is 128, this means that the encoder will output a tensor of size `(n_ent, 256)`, and the score method splits this encoder like so:

   ```python
       def score(self,
                 *,
                 head_embeddings: Tensor,
                 tail_embeddings: Tensor,
                 edge_embeddings: Tensor,
                 **_
                 ) -> Tensor:

        real_head_embedddings, imaginary_head_embeddings = tensor_split(head_embeddings, 2, dim = 1)
        real_tail_embedddings, imaginary_tail_embeddings = tensor_split(tail_embeddings, 2, dim = 1)
        real_edge_embedddings, imaginary_edge_embeddings = tensor_split(edge_embeddings, 2, dim = 1)

        score = (real_head_embedddings * (real_edge_embedddings * real_tail_embedddings + imaginary_edge_embeddings * imaginary_tail_embeddings) + 
                imaginary_head_embeddings * (real_edge_embedddings * imaginary_tail_embeddings - imaginary_edge_embeddings * real_tail_embedddings)).sum(dim = 1)

        return score
   ```

   ```{note}
      Other solutions exist to handle multiple embedding space in the decoder, for example using two different encoders in parallel to encode the two spaces of the same batch. Currently, KGATE does not implement these other methods.
   ```
```
