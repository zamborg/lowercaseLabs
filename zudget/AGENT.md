# AGENT.md

This document summarizes the interactions and decisions made by the Gemini CLI agent during the development of the Zudget application.

## Session Summary

The agent was tasked with building a personal finance application, Zudget, based on a detailed technical specification. The development followed a full-stack approach, including a Python FastAPI backend, a React frontend, and Docker for containerization.

## Key Decisions and Implementations

*   **Backend Setup:**
    *   **Framework:** FastAPI was chosen for the backend due to the user's preference for Python.
    *   **Database:** SQLite with SQLAlchemy was used for data persistence, as requested.
    *   **Containerization:** Docker and Docker Compose were set up to ensure a consistent development environment for both backend and frontend services.
    *   **Data Modeling:** Core data models (User, Account, Transaction, Category, Tag) were implemented using SQLAlchemy.
    *   **CSV Import:** A Python script (`import_csv.py`) was developed to ingest sample transaction data from a CSV file into the database. This script was iteratively refined to handle the actual CSV format and ensure proper database initialization.
    *   **API Endpoints:** Initial API endpoints were created for user management, transaction retrieval, and specialized data for the Sankey diagram (`/sankey/`) and uncategorized transactions (`/transactions/uncategorized`).
    *   **CORS:** `CORSMiddleware` was added to the FastAPI application to resolve cross-origin resource sharing issues between the frontend and backend.

*   **Frontend Setup:**
    *   **Framework:** React was chosen for the frontend, as specified in the technical document.
    *   **Scaffolding:** `create-react-app` was used to quickly set up the React project structure.
    *   **Containerization:** The frontend was integrated into the Docker Compose setup, allowing it to run alongside the backend.
    *   **Routing:** `react-router-dom` was implemented to manage navigation between different UI components.
    *   **UI Components:**
        *   **Transactions List:** A basic table component (`Transactions.js`) was created to display transaction data fetched from the backend.
        *   **Sankey Diagram:** A `Sankey.js` component was developed using `d3` and `d3-sankey` to visualize cash flow. This required careful alignment of data formats between the backend and frontend.
        *   **Tinder-esque Tagging UI:** An interactive swipeable card interface (`TinderUI.js`) was implemented for transaction categorization. Initial attempts with `react-tinder-card` faced compatibility issues with React 19, leading to a pivot to a custom implementation using `@use-gesture/react` and `react-spring` for greater control and compatibility.
    *   **Styling:** Basic CSS was applied to improve the visual presentation of the UI components.

## Troubleshooting and Debugging

*   **Python Import Errors:** Resolved `ImportError` in `import_csv.py` by running it as a module (`python -m app.import_csv`).
*   **Database Table Creation:** Addressed `sqlite3.OperationalError: no such table` by ensuring `models.Base.metadata.create_all(bind=engine)` was called before data import in `import_csv.py`.
*   **Frontend Blank Pages (Sankey):** Identified and fixed an issue where the `d3-sankey` library expected node indices for links, while the backend was providing node names. The `get_sankey_data` function in `crud.py` was updated to return data in the correct format.
*   **Frontend Blank Pages (Tinder UI - `react-tinder-card`):** Encountered significant compatibility issues with `react-tinder-card` and React 19, leading to `TypeError: Cannot read properties of undefined (reading 'div')`. After attempting to resolve with `--legacy-peer-deps` without success, the decision was made to replace `react-tinder-card` with a custom implementation using `@use-gesture/react` and `react-spring`.
*   **CORS Policy Block:** Resolved `Access-Control-Allow-Origin` header error by adding `CORSMiddleware` to the FastAPI application in `main.py`.

## Future Interactions

*   **Feature Development:** The agent is ready to continue implementing features from the technical specification, such as the Rules Engine UI, Amortization workflow, or further enhancements to existing components.
*   **Debugging:** If new issues arise, please provide console logs or error messages to assist in debugging.
*   **Refinement:** The agent can refine existing code, improve styling, or optimize performance as needed.
*   **Testing:** The agent can assist in setting up and running tests for both backend and frontend components.
