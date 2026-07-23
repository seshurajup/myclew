# High-Scoring ONNX Solution Patterns

## Band `18-19`

This band is the broadest optimized-14 high-score tier. The solutions are compact task compilers: they reduce the ARC rule to sparse anchors, small crops, component roles, bitfields, palette facts, or a few scalar measurements, then use a strong final projection to create the only full output canvas.

### Recurring ARC Families

- **Sparse anchors to geometry.** One or two colored cells, edge markers, corners, or separators define a full line, path, frame, cross, diagonal, block placement, or region projection. The high-score version usually avoids explicit object lists and encodes the anchor-to-canvas relation with a fused renderer.
- **Crop, zoom, normalize, and re-render.** Many tasks extract a small active sprite or bbox, normalize it, duplicate it, rotate it, recolor it, or place it back with one terminal op. Useful forms include dynamic bbox crops, top-right or top-left fixed crops, and sampler-based scale/crop transforms.
- **Template transfer and recolor.** A visible motif or reference panel supplies a shape while a separate marker supplies color. These tasks often become color-role embeddings plus direct copy/stamp equations.
- **Periodic, modular, and hidden-patch recovery.** Repeating rows, diagonal residues, bit-packed masks, checkerboards, ring/frame patterns, and missing periodic tiles can be recovered from witnesses instead of discovering a full period explicitly.
- **Local morphology and component roles.** Border/interior detection, hole filling, isolated-pixel edits, local rings, missing corners, and adjacency rules score well when expressed as a single local stencil, grouped convolution, or quantized integer decoder.
- **Separator, panel, and region compression.** Separator-defined grids are compressed to small symbolic outputs or collapsed by overlay, priority, union, XOR, or selected-region rules.

### ONNX Idioms

- **Fused `Einsum` renderers.** Repeated `input` operands encode equality, existence, row/column compatibility, palette transfer, count powers, and final placement without named full-grid masks.
- **Quantized and integer decoders.** `QLinearConv`, `ConvInteger`, `ConvInteger` with dynamic zero points, `ConvInteger` templates, and `ConvInteger` threshold grids are useful when the rule can be expressed as small signed logits.
- **Bit-packed masks.** `UINT8`, `UINT16`, `UINT64`, `BitShift`, `BitwiseAnd`, `BitwiseOr`, and row/column packing are useful for binary connectivity, modular patterns, duplicate flags, and compact periodic state.
- **Dynamic crop and sampling ops.** `Resize`, `RoiAlign`, `MaxRoiPool`, `GridSample`, `MaxPool` indices, and negative/asymmetric pads can combine crop, flip, scale, placement, and clearing.
- **Coordinate moments and scalar geometry.** Row/column vectors, powers, logarithms, exponentials, `ReduceSum`, `TopK`, moments, and sparse scatter infer centers, dimensions, ranks, and selected colors without component enumeration.
- **Sequence and text-like primitives.** `TfIdfVectorizer`, recurrent separator counting, run-length encodings, and sorted color/count vectors are useful when a task is better treated as a compact sequence than a canvas.

### Cost-Saving Rules

- Reduce to a crop, scalar, bitfield, or small code grid early; project to `[1,10,30,30]` only at graph `output`.
- Prefer `BOOL`, `UINT8`, `INT8`, `FLOAT16`, and quantized ops for charged intermediates when exact float tensors are unnecessary.
- Encode colors as scalar IDs, role embeddings, or small basis vectors; reconstruct one-hot only in the terminal op.
- Put geometry in attributes: `pads`, `strides`, `dilations`, `kernel_shape`, `axes`, `equation`, ROI attributes, and old-style slice metadata.
- Spend MACs and a small initializer when it removes named full-grid masks. The scorer charges parameters and activation bytes, not arithmetic.
- Use zero-hot padded rows/cols as blank sources and black channel `0` as real in-grid black when the generator requires it.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task105` | 18.996113 | 405 | Sparse structure to compact renderer. |
| `task165` | 18.991187 | 407 | Geometry-heavy direct synthesis from small role state. |
| `task242` | 18.962129 | 419 | Crop/normalize/re-render family with compact state. |
| `task075` | 18.959745 | 420 | Template transfer and local motif stamping. |
| `task275` | 18.947911 | 425 | Anchor-driven shape completion. |
| `task123` | 18.945561 | 426 | Periodic or sequence extension by fitted rule. |
| `task213` | 18.938543 | 429 | Component-role reconstruction with compact features. |
| `task341` | 18.899681 | 446 | Hidden patch or symmetry recovery. |
| `task218` | 18.811736 | 487 | Separator/block compression to symbolic output. |
| `task230` | 18.868774 | 460 | Local marker insertion by compact stencil. |

### Additional Useful Ideas

- Fitted local convolutions are useful for local generators when the exact symbolic rule is easier to learn as a stencil.
- Huge direct `Einsum` equations are useful when the generator has fixed algebraic structure.
- `GridSample`, `RoiAlign`, `MaxRoiPool`, recurrent counting, bit-packed connectivity, and spectral or component-count tricks are all useful high-score tools when their task family matches.
- Negative pads, sentinel values, underflow thresholds, and signed logits are useful once the final scorer only checks `pred > 0`.

## Band `19-20`

This band keeps cost lower by making the full canvas appear later. The dominant pattern is a small symbolic state - counts, ranks, anchors, panels, tiny sprites, or periodic residues - followed by a direct renderer. Many solutions are exact generator encodings rather than general ARC solvers.

### Recurring ARC Families

- **Select, measure, and redraw objects.** Rarest or largest rectangles, tallest bars, unique sprites, modal colors, and ranked colors are reduced to a few scalar measurements before a small template is rendered.
- **Sparse anchors to paths, rays, and frames.** Dots, edge colors, red/cyan street endpoints, markers, and border hints define bounce paths, axes, plus signs, lines, frames, and one-pixel moves.
- **Small sprite extraction and stamping.** Centered `3x3` sprites, corner patches, red-cell anchored blocks, duplicated motifs, rotations, and compact seen-color grids are handled by gather maps, sampler ops, or tiny direct kernels.
- **Panel, quadrant, and priority overlays.** Fixed separated regions collapse by precedence, union, intersection, XOR, common-black overlap, or color-priority rules.
- **Count, rank, and template synthesis.** Color counts, object counts, bar heights, distinct-color counts, and unary counts become compact outputs such as `1xN` rows, `3x3` patterns, or ranked bars.
- **Periodic, mirrored, and modular reconstruction.** Hidden columns, checker completions, mirrored quadrants, repeated rows, and modular residue patterns become gather tables, bitfields, or one fused equation.
- **Local connectivity and stencils.** Isolated recolor, pair insertion, mini-cell counting, rectangle hollowing, and short-range morphology score well when collapsed to a single stencil or quantized projection.

### ONNX Idioms

- **One heavy op as the program.** `Einsum`, grouped `Conv`, `ConvTranspose`, `ConvInteger`, `QLinearConv`, and `QLinearMatMul` often absorb mask, select, recolor, repeat, and placement.
- **Moments and coordinate bases.** Row/column vectors and small powers infer centers, lengths, dimensions, rarity, and direction without `NonZero`.
- **Dynamic indexing.** `Gather`, `GatherND`, `GatherElements`, `Slice`, `TopK`, `OneHot`, `ScatterND`, and `ScatterElements` are useful for rotations, width doubling, sparse moves, ranked colors, and small crops.
- **Low-byte discrete pipelines.** `BOOL`, `UINT8`, `INT8`, `Mod`, `Sign`, `Hardmax`, bit ops, `QuantizeLinear`, and `MaxPool` indices reduce charged activation size.
- **Direct stencil renderers.** Grouped `Conv`, `ConvInteger`, and `ConvTranspose` handle local insertion, block stamping, checker weaving, and color-preserving copy.
- **Trigonometric, folded, and logarithmic coordinates.** Numeric coordinate encodings are useful when the generator range is controlled and the final sign is all that matters.

### Cost-Saving Rules

- Keep charged tensors as scalars, `[1,10]` color vectors, tiny crops, `3x3` templates, `4x4` overlays, or short rank lists.
- Hard-code generator geometry: known panel sizes, separators, periods, edge positions, `3x3` cells, and fixed symbol layouts.
- Compare direct gather-map cost against dynamic `ConvTranspose` seed cost; either can be best depending on seed length and attributes.
- Use role vectors and low-rank embeddings instead of full channel-pair tables.
- Cast only tiny probes or compact crops; whole-input dtype changes rarely help if they add an intermediate.
- Use padded zero-hot rows or absent generator channels as cheap clearing sources.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task049` | 19.996054 | 149 | Object measurement and analytic rectangle rendering. |
| `task344` | 19.989365 | 150 | Compact symbolic extraction with direct output. |
| `task315` | 19.976119 | 152 | Red-anchor `3x3` motif stamping. |
| `task194` | 19.976119 | 152 | Small fixed sprite transformation. |
| `task094` | 19.969562 | 153 | Local rule compressed into a compact projector. |
| `task108` | 19.950144 | 156 | Sparse anchor to structured output. |
| `task122` | 19.950144 | 156 | Template transfer with small feature state. |
| `task353` | 19.943754 | 157 | One-pixel movement from sparse markers. |
| `task059` | 19.912404 | 162 | Panel or shape rewrite with direct projection. |
| `task072` | 19.912404 | 162 | Split-panel XOR or overlay. |
| `task256` | 19.900134 | 164 | Count or line length to analytic pattern. |
| `task321` | 19.888012 | 166 | Fixed panel collapse by priority. |

### Additional Useful Ideas

- Lookup tables and fixed gather maps are excellent for bounded shapes and known sizes.
- `TopK`, `ArgMax`, and `MaxPool` indices are useful when the generator guarantees a unique winner or fixed tie behavior.
- Dynamic kernels can save memory when they replace color/background mask pipelines.
- High-order repeated `Einsum` is useful for row/column witnesses, palette transfer, and count powers when activation memory would dominate.

## Band `20-21`

This band is mostly direct-output compilation. The useful solutions avoid general object parsing and instead encode the generator as one final `Einsum`, one strided/dilated convolution, one dynamic `ConvTranspose`, or a tiny classifier plus terminal broadcast.

### Recurring ARC Families

- **Small symbolic selection and exemplar extraction.** Fixed crops, glyphs, corner sprites, top-right patches, rotated `3x3` grids, and reference exemplars are copied or transformed by one gather, sampler, or direct equation.
- **Count, rank, and template outputs.** Red counts, distinct-color counts, cup spaces, modal colors, object counts, and bar ranks become canonical `3x3`, `1xN`, or recolored bar outputs.
- **Periodic, modular, and size-aware synthesis.** Diagonal stripes, every-third columns, parity recolor, concentric cycling, row repetition, and dynamic scale use coordinate bases or sampler attributes.
- **Overlay, masked recolor, and region fill.** Split panels, common-black overlap, half comparisons, side-panel union, cyan rectangle completion, and fixed divider overlays collapse into small comparison kernels or direct renderers.
- **Line, span, and column completion.** Red columns, blue/green rows, endpoints, crosses, marker products, downward fills, and priority intersections use row/column reductions inside one terminal op.
- **Marker-to-block and local stencil expansion.** Fixed markers trigger block copies, local insertions, rectangle hollowing, border/interior cleanup, and adjacency edits.
- **Fixed remap, mirror, tiling, and run rewrite.** Small grids mirror into quadrants, repeat into larger grids, or rewrite alternating/parity runs with static basis vectors.

### ONNX Idioms

- **Single final-output `Einsum`.** Combine row/column reductions, priority, color routing, in-bounds masks, and output placement in one contraction.
- **Low-rank fitted bases.** Small channel embeddings, row/column bases, and factorized formulas replace full spatial masks and lookup tables.
- **Quantized projectors.** `QLinearConv`, `ConvInteger`, `MatMulInteger`, and dynamic zero points implement signed comparisons and template decoding in low-byte form.
- **Dynamic-weight `ConvTranspose`.** Use a tiny seed as activation and the ARC tensor as weights for color-preserving stamping, shift, fold, duplicate, or line fill.
- **Grouped and dilated `Conv`.** Dilation compares separated panels; grouping performs per-channel lattice completion, fixed crops, and local rules.
- **Sampler and size ops.** `RoiAlign`, `MaxRoiPool`, `Resize`, `ReduceL2`, `Sqrt`, `Range`, and dynamic `Gather` handle variable square sizes and calibrated crop/scale transforms.
- **Bitfield classifiers and probes.** Compact `GatherElements`, `Split`, `Equal`, `Concat`, `ScatterElements`, static probes, and bitfields solve bounded symbolic grids.

### Cost-Saving Rules

- Make the graph `output` the only large tensor whenever possible.
- Prefer one parameterized `Einsum`, `Conv`, or `ConvTranspose` over a readable multi-node mask pipeline.
- Use signed logits; exact probabilities are unnecessary under final `>0` thresholding.
- Let op attributes carry placement, crop, stride, dilation, group count, resize mode, and equations.
- Use low-byte scalar state only when it reduces a real intermediate body.
- Spend a small static mask or template when it removes several `Slice`, `Pad`, `Where`, or `Concat` outputs.

### Useful Exemplars

| Task | Score | Cost | Pattern |
|---|---:|---:|---|
| `task346` | 20.992667 | 55 | Compact direct-output symbolic renderer. |
| `task395` | 20.974648 | 56 | Common-black overlay by tiny comparison kernels. |
| `task347` | 20.974648 | 56 | Dilated panel overlay with direct projection. |
| `task006` | 20.974648 | 56 | Fixed split-panel intersection. |
| `task139` | 20.939557 | 58 | Small symbolic extraction and direct output. |
| `task203` | 20.905655 | 60 | Template/panel rule compiled to one projector. |
| `task146` | 20.905655 | 60 | Compact stencil or coordinate renderer. |
| `task380` | 20.905655 | 60 | `3x3` rotation by static gather/projection. |
| `task298` | 20.905655 | 60 | Concentric or layer color cycling by `Einsum`. |
| `task254` | 20.905655 | 60 | Bar rank selection and recolor. |
| `task152` | 20.872866 | 62 | `3x3` tile mirrored into quadrants. |
| `task142` | 20.872866 | 62 | Small-grid mirror with direct synthesis. |

### Additional Useful Ideas

- Learned-looking `Einsum` bases and factorized formulas are useful when the generator is controlled enough to make numeric encodings exact.
- `LayerNormalization`, dynamic `Resize`, negative pads, old-style `Slice` attributes, and probe-based glyph decoding are useful special tools.
- Full readable pipelines are useful for deriving rules; optimized versions usually collapse them into direct-output ops.
- Underflow, overflow, sentinel `inf`, and NaN-false behavior are useful sign-control tools after verification.

## Band `21-22`

This band is a parameter-light direct-output regime. Many solutions make one `Gather`, `Einsum`, grouped `Conv`, `ConvInteger`, `RoiAlign`, or dynamic-weight `ConvTranspose` write the final tensor. More complex tasks still fit if all real reasoning is scalar, `1x1`, `3x3`, or a short seed before materialization.

### Recurring ARC Families

- **Dynamic spatial transfer.** Source pixels define colors while a tiny seed defines where they land: checker/weave generation, corner folding, vertical shifts, bottom rectangle lifts, and panel folding.
- **Fixed mirror, remap, and compression.** Whole tasks become one axis `Gather`, grouped strided `Conv`, or direct sampler for vertical mirror stacks, horizontal mirrors, bottom-to-top reflection, and `9x9 -> 3x3` block condensation.
- **Axis, row, and column completion.** All-black row/column recolor, empty interior axes, row/column line fill, endpoint extension, and priority intersections are fused into algebraic row/column renderers.
- **Tiny symbolic classifiers.** Two-cell probes, distinct-color counts, cup-space counts, color-class tests, and binary symmetry tests drive `1x1` answers or small templates.
- **Count/rank/template synthesis.** Nonblack counts, red-box counts, blue counts, modal colors, bar ranks, and self-dot count signatures select canonical rows or templates.
- **Local morphology and symmetry repair.** Rectangle hollowing, local support cleanup, yellow cutout restoration, periodic black-cutout repair, and divider folding use dynamic `ConvTranspose` or high-order relational `Einsum`.
- **ROI and sampler extraction.** Duplicate-half extraction and magnified linegrid extraction use scalar orientation/size inference followed by `RoiAlign`.

### ONNX Idioms

- **Input-as-weight `ConvTranspose`.** `ConvTranspose(seed, input) -> output` preserves arbitrary colors while attributes and seed taps encode shift, fold, duplicate, fill, mirror, or crop.
- **Signed and sentinel seed taps.** Positive taps paint, negative taps suppress, and `inf` or `-inf` can make spill become NaN-false under the final threshold.
- **Single terminal `Einsum`.** Repeated inputs encode row/column occupancy, equality, mode, count powers, compatibility witnesses, hole detection, and output-channel decisions.
- **Static `Gather` maps.** A 30-entry row or column vector can encode mirror/copy plus zero-hot sentinel padding in one node.
- **Grouped `Conv` and `ConvInteger` templates.** Grouped kernels compress uniform blocks; `ConvInteger` threshold grids decode scalar counts into black/color logits.
- **ROI as dynamic cropper.** `RoiAlign` can encode crop, flip, downsample, and duplicate-half extraction after only scalar size or orientation inference.
- **Tiny role embeddings.** `10x2`, `10x3`, or length-10 role vectors carry color behavior, black suppression, placeholder underflow, and output routing.

### Cost-Saving Rules

- Name the heavy renderer `output`; output memory is excluded.
- Keep branch state scalar and let one final op select the branch.
- Store color logic as low-rank embeddings or role vectors rather than channel-spatial masks.
- Use padded zero-hot rows/cols as clearing sentinels in gathers and samplers.
- Prefer a short `ConvTranspose` seed when attributes can encode a mirror or shift more cheaply than a full gather index.
- Use direct signed logits, underflow, and bias instead of exact one-hot construction.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task082` | 21.955478 | 21 | Dynamic `ConvTranspose` renders top-row seeds into a checker weave. |
| `task047` | 21.908958 | 22 | One `Einsum` extends row/column axes and paints red intersections. |
| `task303` | 21.821946 | 24 | All-black row/column recolor by shared-matrix `Einsum`. |
| `task186` | 21.821946 | 24 | Blue count to red template through scalar self-dot and `ConvInteger`. |
| `task299` | 21.821946 | 24 | Direct `Einsum` completes red/cyan streets and yellow crossing. |
| `task399` | 21.781124 | 25 | Scalar count hash plus `ConvInteger` template decoder. |
| `task296` | 21.667795 | 28 | Corner folding by dynamic `ConvTranspose` and spill suppression. |
| `task056` | 21.667795 | 28 | Tiny two-probe classifier to `1x1` color. |
| `task063` | 21.667795 | 28 | Free row/column fill by orientation-invariant `Einsum`. |
| `task164` | 21.598803 | 30 | One `Gather(axis=3)` mirrors `3x3` into `3x6`. |
| `task116` | 21.598803 | 30 | One `Gather(axis=2)` creates a vertical mirror stack. |
| `task172` | 21.598803 | 30 | One row gather duplicates and mirrors `3x3` input. |

### Additional Useful Ideas

- Dynamic-weight `ConvTranspose` is a strong copier whenever colors should pass through unchanged.
- Repeated-input `Einsum` is useful for `exists`, equality, witness-copy, count powers, row/column compatibility, and mode selection.
- `Shrink`, `Mod`, `QuantizeLinear`, `Cast(float -> UINT8)`, and dynamic `ConvInteger` zero points are useful small integer tricks.
- `ReduceL2` on one-hot padded inputs can recover square side length when all true cells are occupied.
- Direct samplers can replace explicit width/height detection when the generator bounds are tight.

## Band `22-23`

This band is almost entirely one-node or one-terminal-op logic with a tiny vector, seed, bias, or embedding. Pure palette maps score at cost `10`; nontrivial relational rules still fit when compressed into direct `Einsum`, `Conv`, or dynamic `ConvTranspose`.

### Recurring ARC Families

- **Pure palette remap.** Same geometry, only channels move. Examples include fixed color involutions, gray/cyan swaps, orange-to-gray remaps, magenta-to-red, and sparse recolors under generator color assumptions.
- **Pattern repair from visible witnesses.** Periodic stripes, modular tables, and black cutouts are restored by row/column compatibility rather than explicit period detection.
- **Guide-color and row-template propagation.** Left-edge colors, full reference rows, placeholders, or sparse row witnesses fill row patterns while black/background is preserved.
- **Tiny fixed-grid spatial generators.** Checkerboard bars, downward ray fills, marker drops, macro-grid completion, small mirror/concatenate, and fixed `3x3` or `5+divider+5` overlays are encoded by short `ConvTranspose` seeds.
- **Marker-driven recolor and coordinate products.** Corner marker colors recolor a fixed object; top-row and right-edge markers produce red Cartesian intersections.
- **One-bit classification.** Symmetry or equality reduces to one scalar logit and a tiny answer painter.
- **Rectangle completion.** Row and column support fill ragged colored strips into a rectangle while preserving existing color.

### ONNX Idioms

- **`Gather(axis=1)` palette table.** A length-10 output-channel-to-input-channel vector is the canonical recolor op. Unused output channels can gather an absent generator channel to clear.
- **Relational `Einsum` as rule compiler.** Repeated `input` operands implement row witnesses, column witnesses, anchors, row histograms, marker extraction, source-color transfer, and output-channel routing in one terminal contraction.
- **Length-10 role vectors.** One vector can mark black negative, placeholders tiny, anchor colors strong, valid colors positive, and nonblack witnesses.
- **Low-rank color embeddings.** `10x2` or similar codes can detect roles and decode output channels inside a high-rank equation.
- **Dynamic-weight `ConvTranspose`.** A short seed plus `input` as weights encodes checker parity, fixed mirrors, panel overlay, downward fill, marker drops, and macro-cell completion.
- **Self-correlation `Conv(input, input)`.** The input can be both data and kernel to count aligned one-hot matches for scalar classification.
- **Signed, tiny, and sentinel logits.** Positive target channels are enough; zero, negative, underflowed, `inf`/`-inf`, or NaN-false values suppress alternatives.

### Cost-Saving Rules

- Emit directly to graph `output`; a helper mask usually costs too much in this band.
- Prefer 10-entry vectors, 3-value seeds, 16-tap seeds, or one compact embedding table over spatial constants.
- Put geometry in `axis`, `equation`, `pads`, `dilations`, `strides`, `kernel_shape`, and `output_shape`.
- Compare a gather index against a `ConvTranspose` seed; seed plus attributes can beat a 30-entry map.
- Use existing float one-hot input directly when there are no charged intermediates.
- Let high-MAC contractions replace row/column masks, period detection, or bbox extraction.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task373` | 22.920558 | 8 | One `ConvTranspose` seed renders fixed two-row bars into checkerboard. |
| `task016` | 22.697415 | 10 | One `Gather(axis=1)` color-negative palette transform. |
| `task337` | 22.697415 | 10 | Exact gray/cyan channel swap by `Gather`. |
| `task276` | 22.697415 | 10 | Sparse magenta-to-red/orange-preserve recolor. |
| `task309` | 22.697415 | 10 | Orange-to-gray remap by channel lookup. |
| `task305` | 22.697415 | 10 | Diagonal-period stripe repair by witness `Einsum`. |
| `task061` | 22.697415 | 10 | Modular table cutout completion by row/column witnesses. |
| `task267` | 22.697415 | 10 | Marker color extraction and object recolor in one `Einsum`. |
| `task017` | 22.697415 | 10 | Periodic field repair from visible witnesses. |
| `task197` | 22.697415 | 10 | Row-template completion from full reference row. |
| `task372` | 22.435051 | 13 | Fixed panel overlay with dynamic `ConvTranspose`. |
| `task103` | 22.291950 | 15 | Scalar self-`Conv` symmetry classifier to `1x1` answer. |
| `task073` | 22.227411 | 16 | Marker fall by `ConvTranspose` seed and channel bias. |
| `task166` | 22.004268 | 20 | Cyan rectangle completion by one low-rank `Einsum`. |
| `task043` | 22.004268 | 20 | Edge-marker Cartesian product with red intersections. |

### Additional Useful Ideas

- A distinctive visible color can anchor modular alignment without computing the period.
- Placeholder channels can be numerically present as gates but too tiny to win the output channel.
- Bias vectors are useful when they replace thresholding nodes or black/background masks.
- Row-count times column-count is a compact replacement for bbox extraction when the generator guarantees rectangular completion.
- `ConvTranspose` can act as a finite-difference classifier, overlay merger, mirror generator, or checker renderer, not only as an upsampler.

## Band `23-24`

Optimized-14 has 2 tasks in this band. Both are pure grid-enhancement tasks: preserve one-hot colors, enlarge each source cell into a fixed block, and let zero-hot padding define the area outside the true output. Scores are determined almost entirely by initializer count: cost `4` gives `23.613706`, cost `5` gives `23.390562`.

### Recurring ARC Families

- **Grid enhancement and nearest block upsampling.** `task307` scales variable square inputs `N x N`, `N=2..5`, to `2N x 2N`; `task223` scales fixed `3x3` inputs to `9x9`.
- **Color-preserving spatial resampling.** No palette inference is needed. Black inside the true grid remains channel-0 black; outside-grid cells stay zero-hot.

### ONNX Idioms

- **Dynamic-weight `ConvTranspose` block stamper.** `task307` uses `ConvTranspose(["D", "input"], ["output"])` with `D=ones((1,1,2,2))` and the live ARC tensor as weights. `dilations=[2,2]` place source pixels on doubled coordinates and the `2x2` seed fills each block.
- **`MaxRoiPool` as block upsampler.** `task223` uses one ROI `[[0,0,0,9,9]]`, `pooled_shape=[30,30]`, and `spatial_scale=1.0` to repeat each `3x3` source cell into a `3x3` output block while sampling zero-hot padding after the true `9x9` output.
- **One-hot transport.** Both solutions move all 10 channels directly, avoiding `ArgMax`, scalar color IDs, equality tests, recolor tables, and final reconstruction.
- **Terminal materialization.** The only node writes graph `output`, so activation memory is avoided.

### Cost-Saving Rules

- For block upsampling, try direct-output spatial ops before `Resize`, `Tile`, `Gather`, `Slice`, `Pad`, or mask-based expansion.
- Choose the expansion primitive by parameter count: 2x upsampling is cheaper as a `2x2` seed (`c4`), while fixed 3x enhancement is cheaper as one 5-value ROI (`c5`) than a `3x3` seed.
- Encode scale, placement, and crop in `dilations`, `pads`, `pooled_shape`, and `spatial_scale`.
- Let source zero-hot padding clear output rows/cols beyond the enhanced grid instead of detecting true height or width.
- Keep input/output as `FLOAT`; casts do not help when there are no charged intermediates.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task307` | 23.613706 | 4 | Dynamic-weight `ConvTranspose` for variable `2x` grid enhancement. |
| `task223` | 23.390562 | 5 | One `MaxRoiPool` for fixed `3x3 -> 9x9` enhancement. |

### Additional Useful Ideas

- Dynamic-weight `ConvTranspose` with an all-ones seed is a nearest-neighbor upsampler that preserves arbitrary colors.
- ROI pooling is useful when a fixed enlargement can be expressed with one ROI whose five parameters beat a larger seed or gather table.
- For crop-plus-expand tasks, combine asymmetric pads, negative pads, strides, and dilations so crop, placement, and enlargement happen in the terminal node.
- Initializer dtype does not reduce parameter cost here; the scorer counts elements, and output tensor memory is excluded.

## Exact `25`

Optimized-14 has 9 exact-score tasks. Most are single-node, zero-initializer spatial operators where the entire rule lives in attributes. One task uses a single scalar constant with repeated-input `Einsum` and float underflow to select the exact mode color.

### Recurring ARC Families

- **Fixed spatial transforms.** Diagonal reflection, 180-degree rotation, fixed shifts, and fixed crops can run on the full padded `[1,10,30,30]` tensor with no size detection.
- **Attribute-only samplers.** `Transpose`, `MaxPool`, and `LpPool` encode crop, reverse, shift, pad, dilation, and placement without initializer elements.
- **Fixed patch extraction.** Top-left `2x2`, top-right `3x3`, and similar fixed crops are solved by pool windows that sample real cells for the output prefix and zero-hot padding afterward.
- **Repeated square tile extraction.** If the desired tile is already the left panel, copy it and derive the dynamic width mask from row occupancy with a parameter-free `Einsum`.
- **Mode-color fill.** A `3x3` exact-count rule can be solved by count powers and underflow, then masked by the in-bounds input.
- **Color-preserving geometry.** One-hot channels carry all colors through spatial movement; no label decode is needed.

### ONNX Idioms

- **Terminal `Transpose`.** `Transpose(input -> output, perm=[0,1,3,2])` solves diagonal reflection for fixed and variable square grids at cost `0`.
- **Terminal `LpPool` as reverse or crop sampler.** `kernel_shape=[1,1]` with negative strides/pads handles fixed 180-degree rotation; larger dilated windows handle fixed top-right crops when only one real tap survives.
- **Terminal `MaxPool` as crop or row sampler.** Large dilation, negative strides, and negative pads can route selected rows/cols or crop prefixes while later output cells read outside the source.
- **Parameter-free `Einsum` masks.** `Einsum("nchw,ndwk->nchw")` copies the left panel and uses row occupancy relabeled as output width to mask columns beyond the true square.
- **One-scalar underflow `Einsum`.** Repeating `input` many times computes `count(color)^K`; one calibrated scalar makes smaller counts underflow to zero while the desired exact count remains positive.
- **Zero-hot padding as shape signal.** Padded rows/cols both clear outside output and provide dynamic extent masks without `Shape`, `Range`, or comparisons.

### Cost-Saving Rules

- First try `Transpose`, `LpPool`, and `MaxPool` attributes for fixed geometric movement before adding any constants.
- Emit directly to graph `output`; a single terminal node with no initializers is exact `25`.
- Avoid `Slice`, `Pad`, `Gather`, size inference, casts, masks, and decode tables when a full-canvas spatial op preserves one-hot channels.
- Use known in-grid black cells for real black output and zero-hot padded cells for outside-grid clearing.
- Replace tiny-seed dynamic `ConvTranspose` samplers with attribute-programmed pool samplers when each output cell has at most one valid source.
- For exact-cardinality tasks, high-MAC repeated `Einsum` plus underflow can beat histograms, comparisons, and `TopK`.

### Useful Exemplars

| Task | Score | Cost | Pattern |
| `task179` | 25.000000 | 0 | Fixed `3x3` main-diagonal reflection by terminal `Transpose`. |
| `task241` | 25.000000 | 0 | Variable square diagonal reflection by the same `Transpose`. |
| `task087` | 25.000000 | 0 | Fixed `3x3` rotate-180 by `LpPool` negative strides/pads. |
| `task140` | 25.000000 | 0 | Another zero-initializer `LpPool` rotate-180 sampler. |
| `task326` | 25.000000 | 0 | Top-left `2x2` extraction by attribute-only `MaxPool`. |
| `task135` | 25.000000 | 0 | Fixed `9x9 -> 3x3` top-right crop by `LpPool`. |
| `task053` | 25.000000 | 0 | Fixed `3x3` downward shift by `MaxPool` row sampler. |
| `task067` | 25.000000 | 0 | Repeated-tile crop by parameter-free `Einsum` extent mask. |
| `task129` | 25.000000 | 1 | Exact mode-color fill by repeated-input underflow `Einsum`. |

### Additional Useful Ideas

- Pooling attributes can encode nonstandard sampling routes while preserving all 10 one-hot channels.
- `LpPool` can be a direct-output crop-and-place operator, not just a pooling layer, when dilation separates candidate taps so only one in-grid tap survives.
- `MaxPool` with large dilation is useful when a crop or shift can be expressed as one valid tap and the rest out of bounds.
- Axis relabeling inside `Einsum` can replace transpose, gather, shape, and comparison logic when one spatial dimension gates another.
- If a repeated panel already equals the target, copy that panel and mask extents; comparing panels is unnecessary.
- Recheck near-exact shift, crop, flip, and sampler solutions for replacement by attribute-only `Transpose`, `LpPool`, or `MaxPool`.
