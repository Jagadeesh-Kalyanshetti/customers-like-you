# ML1: Method decision (written before fitting any model)

Day 6, 2026-10-13. I have not fitted anything yet.

1. I choose Truncated SVD as my main model and will run PCA with the same k in ML5.
2. My matrix is 97.95% empty and takes 4.99 MB stored sparse. Truncated SVD works on it as it is.
3. PCA subtracts each column's average first. That turns the zeros into small negatives, so the matrix would no longer be sparse (161.43 MB if written out dense).
4. PCA tells me how each customer differs from the average customer, but SVD tells me which products are bought together by the same customers.
5. "Customers who bought this also bought" is about what people buy together, not about differences from an average customer, so SVD fits the question better.
6. One SVD fit gives customer factors and product factors together, and I need product factors for the recommender (brief Section 3.3).
8. This is a prediction, not a result. In ML5 I will compare PCA and SVD further.