<!--May need to be merged with data_leakage-->
# The KGATE preprocessing workflow

**KGATE** is more than just a library. It is meant to provide users with a guided framework and lots of automated routines. Among them, the one you will run into the most often is the preprocessing workflow, what is called when you create an `Architect` object with a raw **knowledge graph**.

```{note}
Preprocessing your **knowledge graph** with **KGATE** is highly recommended, but not mandatory. You can also submit your custom-split KG as long as it already is a tuple `KnowledgeGraph` objects (static methods to convert widely used format to **KGATE** format may help you with that), or a compatible pickle file. In this case, preprocessing will be ignored.
```

All steps of the preprocessing workflow can be omitted or customised. See the [configuration file]() for all the parameters available.

## Knowledge Graph parsing

The **knowledge graph** can be given as a csv file with at least the columns `from` (head nodes), `rel` (relation nodes) and `to` (tail nodes), or as a python object compatible with KGATE. Currently, compatible objects are [TorchKGE's KnowledgeGraph](https://torchkge.readthedocs.io/en/latest/reference/data.html#knowledge-graph) and [PyTorch Geometric's HeteroData](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData).

If a data structure is not supported yet, you can ask for it to be added or implement it yourself and submit a pull request on the [repository](https://github.com/BAUDOTlab/KGATE).

## Filtering out duplicate triples

*related parameter: preprocessing.remove_duplicate_triples*

Sometimes, most commonly when a **knowledge graph** is built using an automated system, the exact same triple *(a,r,b)*. The presence of duplicates brings a lot of possible biases, so they are detected and entirely removed from the dataset.

## Filtering out reverse triples

*related parameters: preprocessing.flag_near_duplicate_relations

For every relation, if a triple *(a,r,b)* also has its reverse *(b,r,a)* in the **knowledge graph**, the second instance is flagged and taken out of the dataset. Indeed, some models cannot handle symmetric relationships, and the reverse relation might have a slightly different semantic meaning.

```{note}
No data is ever truly removed. In reality, these triples are stored in the `knowledge_graph.removed_triples` attribute, but they are not used at all during training. However, they are considered true information when running the inference module on the knowledge graph. The original knowledge graph (including all flagged triples) can be recovered at any time using the `utils.merge_kg()` function.
```

## Data leakage control procedure

Next, the automated **data leakage** checks run sequentially to ensure the best learning condition for the models. See the [Data Leakage](./data_leakage.md) page for more detailed information on the procedure.

## Adding directionality

Most **knowledge graph embedding models** need a fully directed graph to operate properly. However, relations in biomedical knowledge graphs are often undirected. The best example is the PPI relation, indicating an interaction of any kind between two proteins. When an undirected relation is only present as a triplet *(a,r,b)*, **KGATE** will create the reverse relation *(b,r_inv,a)*, where *r_inv* has the same semantical meaning as *r*.

 [Filtering out reverse triples](#filtering-out-reverse-triples)