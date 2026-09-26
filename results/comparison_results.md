# Chunking Strategy Comparison Results

Tested 6 questions across 5 chunking methods.

## Summary

| Method | Chunks Created | Embed Time (s) | Retrieval Hits | Answer Hits |
|---|---|---|---|---|
| fixed_size_chunking | 81 | 174.92 | 4/6 | 5/6 |
| sentence_size_chunking | 65 | 139.87 | 4/6 | 5/6 |
| paragraph_size_chunking | 50 | 108.35 | 3/6 | 5/6 |
| retrival_chunking | 105 | 225.14 | 5/6 | 5/6 |
| syntex_retrival_chunking | 113 | 242.44 | 3/6 | 5/6 |

## Detailed Results

### fixed_size_chunking

| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |
|---|---|---|---|---|
| Which planet is the largest in our solar system? | False | True | 6.84 | Ganymede, one of Jupiter's moons, is the largest moon in the Solar System, larger than the planet Me... |
| What is Olympus Mons? | True | True | 6.71 | Olympus Mons is the largest volcano and second-highest known mountain in the Solar System, located o... |
| How long does Mercury take to orbit the Sun? | True | True | 4.88 | Mercury takes 87.97 Earth days to orbit the Sun.... |
| What is the Great Red Spot? | True | True | 4.88 | Jupiter's iconic Great Red Spot is a persistent anticyclonic storm larger than Earth.... |
| Which planet rotates on its side? | True | True | 4.8 | Uranus rotates on its side.... |
| What is Mount Everest? | False | False | 4.8 | The context provided does not contain any information about Mount Everest. Therefore, I could not fi... |

### sentence_size_chunking

| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |
|---|---|---|---|---|
| Which planet is the largest in our solar system? | True | True | 5.15 | Based on the provided context, Jupiter is the largest planet in our solar system. The context states... |
| What is Olympus Mons? | True | True | 4.77 | Olympus Mons is the largest volcano and second-highest known mountain in the Solar System, located o... |
| How long does Mercury take to orbit the Sun? | True | True | 4.72 | Mercury takes 87.97 Earth days to orbit the Sun.... |
| What is the Great Red Spot? | False | True | 4.85 | The Great Red Spot is an iconic anticyclonic storm larger than Earth located on the planet Jupiter.... |
| Which planet rotates on its side? | True | True | 5.64 | Based on the provided context, there is no information about any planet rotating on its side. The co... |
| What is Mount Everest? | False | False | 4.45 | I could not find the answer.... |

### paragraph_size_chunking

| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |
|---|---|---|---|---|
| Which planet is the largest in our solar system? | True | True | 6.03 | The context does not specify which planet is the largest in our solar system. It provides informatio... |
| What is Olympus Mons? | True | True | 4.88 | Olympus Mons is the largest volcano and second-highest known mountain in the Solar System, located o... |
| How long does Mercury take to orbit the Sun? | True | True | 4.89 | Mercury takes 87.97 Earth days to orbit the Sun.... |
| What is the Great Red Spot? | False | True | 4.84 | The Great Red Spot is a persistent anticyclonic storm larger than Earth, located on the planet Jupit... |
| Which planet rotates on its side? | False | True | 4.6 | Uranus rotates on its side.... |
| What is Mount Everest? | False | False | 4.72 | The context provided does not contain any information about Mount Everest. Therefore, I could not fi... |

### retrival_chunking

| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |
|---|---|---|---|---|
| Which planet is the largest in our solar system? | True | True | 4.77 | Jupiter is the largest planet in our solar system.... |
| What is Olympus Mons? | True | True | 4.77 | Olympus Mons is the largest volcano and second-highest known mountain in the Solar System, located o... |
| How long does Mercury take to orbit the Sun? | True | True | 4.75 | Mercury takes 87.97 Earth days to orbit the Sun.... |
| What is the Great Red Spot? | True | True | 4.85 | The Great Red Spot is an iconic persistent anticyclonic storm larger than Earth found on Jupiter.... |
| Which planet rotates on its side? | True | True | 4.6 | Uranus rotates on its side.... |
| What is Mount Everest? | False | False | 4.49 | I could not find the answer.... |

### syntex_retrival_chunking

| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |
|---|---|---|---|---|
| Which planet is the largest in our solar system? | False | True | 4.87 | Jupiter is the largest planet in our solar system, with a diameter about eleven times that of Earth.... |
| What is Olympus Mons? | True | True | 4.87 | Olympus Mons is the largest volcano and the second-highest known mountain in the Solar System, locat... |
| How long does Mercury take to orbit the Sun? | True | True | 4.73 | Mercury takes 87.97 Earth days to orbit the Sun.... |
| What is the Great Red Spot? | False | True | 5.06 | The Great Red Spot is an iconic anticyclonic storm larger than Earth located on the planet Jupiter. ... |
| Which planet rotates on its side? | True | True | 4.61 | Uranus rotates on its side.... |
| What is Mount Everest? | False | False | 4.74 | The context provided does not contain any information about Mount Everest. Therefore, I could not fi... |
