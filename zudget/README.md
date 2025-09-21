# Zudget

Zudget is a personal finance application with a focus on cashflow analysis and AI-assisted transaction categorization.

## Development

To run the application locally, you will need Docker and Docker Compose.

1.  **Build and run the containers:**
    ```bash
    docker-compose up --build -d
    ```
    This command will build and start both the backend (FastAPI) and frontend (React) services in detached mode.

2.  **Import Sample Data:**
    The application comes with a sample `transactions.csv` file in the `data/` directory. To import this data into the SQLite database:
    ```bash
    docker-compose run --rm backend python -m app.import_csv
    ```
    This command will create the necessary database tables and populate them with the sample transaction data.

3.  **Access the Application:**
    *   **Frontend (UI):** The React application will be available at `http://localhost:3000`.
    *   **Backend API:** The FastAPI backend API will be available at `http://localhost:8000`. You can access the OpenAPI documentation (Swagger UI) at `http://localhost:8000/docs`.

## Implemented Features

*   **Backend:**
    *   FastAPI application with SQLAlchemy and SQLite.
    *   Dockerized development environment.
    *   CRUD operations for Users, Accounts, and Transactions.
    *   CSV import functionality for transaction data.
    *   CORS enabled for frontend communication.
    *   Sankey diagram data endpoint (`/sankey/`).
    *   Uncategorized transactions endpoint (`/transactions/uncategorized`).
*   **Frontend:**
    *   React application with `react-router-dom` for navigation.
    *   **Transactions List:** Displays a table of all transactions.
    *   **Sankey Diagram:** Visualizes cash flow using D3.js.
    *   **Tinder-esque Tagging UI:** A swipeable card interface for categorizing transactions (uses `@use-gesture/react` and `react-spring`).

## Project Structure

```
.
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── crud.py
│       ├── database.py
│       ├── import_csv.py
│       ├── main.py
│       ├── models.py
│       └── schemas.py
├── data/
│   └── transactions.csv
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── public/
    ├── src/
    │   ├── App.css
    │   ├── App.js
    │   ├── components/
    │   │   ├── Sankey.js
    │   │   ├── TinderUI.js
    │   │   └── Transactions.js
    │   └── index.js
    └── ... (other create-react-app files)
```