## Horizon Overview

Horizon repairs data errors by analyzing functional dependency (FD) constraints. Its core components include a graph data structure, FD pattern graph construction, strongly-connected-component (SCC) analysis, topological sorting, and a pattern-based dirty-data repair procedure.

## File Structure

### 1. `graph.py`
- **`Vertex` class**: Represents a vertex in the graph, typically an attribute or an attribute-value pair. It stores the attribute identifier, type (restricted or free), and edges to neighbor vertices with their weights and quality scores.
    - **Methods**:
      - `addNeighbor(nbr)`: Adds an edge from the current vertex to a neighbor.
      - `getConnections()`: Returns all neighbor vertices.
      - `getWeight(nbr)`: Returns the edge weight to the specified neighbor.
- **`Graph` class**: Represents the FD pattern graph, where vertices stand for attributes or attribute values and edges encode FD relationships.
    - **Methods**:
      - `addVertex(key, key1, type)`: Adds a new vertex.
      - `addEdge(f, t)`: Adds an edge from vertex `f` to vertex `t`.
      - `getVertices()`: Returns the IDs of all vertices.
- **Helper functions**:
  - `tr(G)`: Computes the transpose of the graph (reverses all edge directions).
  - `topoSort(G)`: Performs a topological sort on the graph.
  - `walk(G, s)`: Traverses the graph starting from a given vertex and returns the traversal path.

### 2. `horizon.py`
- **`BuildFDPatternGraph`**: Builds the FD pattern graph from the data file and FD constraint file. This graph encodes the dependencies between attributes.
- **`ComputePatternQulity`**: Computes the quality of each pattern in the graph, including support and edge weight.
- **`BuildSCCGraghAndSort`**: Builds the SCC graph of the FD graph and performs topological sorting. Returns the sort order, the vertex-to-component mapping, and the list of components.
- **`OrderFDs`**: Orders the FDs based on SCC structure and topological order.
- **`GeneratePatternPreservingRepairs`**: Repairs dirty data according to the FD graph and ordering, producing pattern-preserving repair results.
- **`dirty_cells`**: Identifies dirty cells between the dirty and clean data files.
- **`Horizon`**: Main entry function — runs the end-to-end cleaning pipeline and returns repaired pattern expressions.

### 3. `util.py`
- **`check_string(string)`**: Checks whether a string contains specific error markers and returns the corresponding error type.
- **`calF1(precision, recall)`**: Computes the F1 score.
- **`calRepPrec(pattern_expressions, dirty_path, clean_path)`**: Computes repair precision.
- **`calRepRec(pattern_expressions, dirty_path, clean_path)`**: Computes repair recall.

## Pipeline

1. **Build the FD pattern graph**:
   - Call `BuildFDPatternGraph` to read the data file and FD constraints and construct the graph object.
   - Each vertex represents an attribute or attribute value; each edge encodes an FD relationship.

2. **Compute pattern quality**:
   - `ComputePatternQulity` performs a depth-first traversal of the graph and computes the support and quality score of each pattern.

3. **SCC analysis**:
   - `BuildSCCGraghAndSort` identifies the strongly connected components and topologically sorts them.

4. **Order the FDs**:
   - Call `OrderFDs` to sort the FDs so the repair order respects dependencies.

5. **Repair dirty data**:
   - Call `GeneratePatternPreservingRepairs` to repair dirty cells according to the ordered FDs and produce pattern-preserving repair expressions.
   - The `Horizon` entry function orchestrates the pipeline and returns the final pattern expressions.

## Key Capabilities

- **Pattern-preserving repair**: Fixes dirty cells by honoring FD constraints.
- **Topological ordering**: Guarantees that repairs are applied in an order consistent with the FD structure.
- **Quality scoring**: Evaluates repair patterns through support and connection quality.
