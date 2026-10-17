# DevSoc AI/ML assignment

There are **two AI/ML tasks**, rather than three independent projects:

1. Read the 12 questions in `text.txt`, call an LLM API and save the returned answers as JSON.
2. Complete the house-price linear regression notebook using NumPy gradient descent. The paper gradient derivation belongs to this task.

## Files and status

- `linear_regression_completed.ipynb`: a simpler implementation in the original 47-cell exercise. All 19 code cells are completed, with short written answers and saved outputs. Includes exploration, encoding, scaling, predictions, MSE, gradients, training, the loss graph and test examples.
- `linear_regression.py`: optional script version of the same calculations; this is not a third task.
- `llm_answers.py` and `text.txt`: API task code and the exact supplied questions.
- `responses.json`: included from the user's successful live Groq run. All 12 original questions are paired with nonempty returned answers. The received answer text has been preserved unchanged.
- `data/`: reference copies of the supplied CSV files. The notebook and script load directly from GitHub, as the assignment specifies.

## Regression

Upload the notebook to Colab, save a copy in your Drive and run all cells. Both datasets are loaded directly from the assignment's GitHub URLs; an internet connection is required. Enable reviewer viewing access on the saved copy and include its link in your repository README. The notebook now completes the original scaffold directly, so no manual transfer is needed. The original instruction says to derive the gradients on paper before implementing them; do that step yourself. It does not explicitly request a photo upload.

Alternatively, from this folder with Python installed:

```text
python -m pip install -r requirements.txt
python linear_regression.py
```

The model has eight weights and one bias. It uses full MSE, hence the factor 2 in both gradients. Feature and target scaling use only training statistics. Missing test category columns are added with a basic loop, and the test table is put in training column order. The supplied CSVs have no missing values; their counts are displayed. No general missing-value pipeline is added.

The simpler version uses `np.random.seed(42)` and `np.random.randn` for initial weights, three short functions and one training loop. Extra baseline calculations, comparison tables, the second plot and diagnostic assertions were removed. Checked results with the supplied datasets: training MSE about 0.00390546 and test MSE about 0.00573100, both on scaled prices. The local verification used the reference CSV copies because the earlier direct-download attempt failed; the delivered code still uses only the assignment's GitHub URLs. Run the uploaded copy in Colab before submitting.

Names follow the DevSoc starter: `x`, `y`, `w`, `b`, `mse_loss_fn`, `get_gradients`, `update`, `NUM_EPOCHS`, and `LEARNING_RATE`. Weights have shape `(1, 8)`, so predictions use `x @ w.T + b`. Andrew Ng's labs use closely related notation, but often use half-MSE and differently shaped arrays. Do not mix their gradient factors or array shapes into this full-MSE implementation without adapting them.

## API task

```text
python -m pip install -r requirements.txt
python llm_answers.py
```

Run the commands from this folder, where `text.txt` is located. Enter your own Groq key at the hidden prompt. The script uses a basic question loop, requests one concise paragraph per question and saves question/answer dictionaries after each success. It rejects responses that do not finish normally, including token-limit cutoffs. An API failure raises the SDK's normal error and stops the program; completed answers remain saved. **Rerunning starts from question one and replaces the previous output after the first success.** A partial file is not a completed submission: check that all 12 original questions have nonempty received answers and review their content.

The script uses Groq's `openai/gpt-oss-20b` with low reasoning effort and saves only the final answer. See the [model list](https://console.groq.com/docs/models) and [reasoning parameters](https://console.groq.com/docs/reasoning), checked 17 October 2026. The included JSON is from the user's successful live run with these same request settings. The script was subsequently simplified without changing the question messages, model or generation settings. The saved real output has been preserved unchanged. Keep the key out of the submission.

## Before submitting

1. API output is generated and checked for 12 matching questions, nonempty answers and complete endings. Read the answers yourself before submitting; model output is not a guarantee of factual correctness.
2. Save and run the notebook in your own Colab/Drive copy, and check reviewer access.
3. Write out the paper derivation requested in the original notebook.
4. Put the code, input questions, real `responses.json`, completed notebook, data and run instructions in an accessible GitHub repository. Add your saved Colab link to its README. Keep API keys out of the repository.
5. Submit the GitHub link through the [AI/ML crew module 0 26-27 form](https://docs.google.com/forms/d/e/1FAIpQLSe9H1urQUnUlDGCmfAEQufUug4QanGGFcJI1ibpzZgOA1p_ug/viewform). Checked 17 October 2026: the form asks for Name (required), BITS ID (optional), Phone Number (required) and GitHub Link (required). Nothing has been submitted for you. The assignment page lists 18 October as the deadline, without a cutoff time.
6. Read the code and practise explaining the loop, JSON records, matrix shapes, scaling and gradients. Do not claim independent authorship of assisted work.

Sources: [assignment brief](https://devsocgoa.notion.site/DevSoc-Crew-Assignment-2026-2027-3d90442f848980df92fbd59ee4d28df3), [original notebook](https://colab.research.google.com/drive/1oi-qfV612KKsnoD_ev16UldNgW_iKzQ8?usp=sharing), [datasets](https://github.com/cronan03/DevSoc_AI-ML), [Groq quickstart](https://console.groq.com/docs/quickstart).

AI assistance was used for implementation, simplification and explanations. Earlier reference code was simplified into these copies; the original files were preserved.
