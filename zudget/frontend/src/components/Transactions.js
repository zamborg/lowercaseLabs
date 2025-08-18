import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Transactions = () => {
    const [transactions, setTransactions] = useState([]);

    useEffect(() => {
        axios.get('http://localhost:8000/transactions/')
            .then(response => {
                setTransactions(response.data);
            })
            .catch(error => {
                console.error('There was an error fetching the transactions!', error);
            });
    }, []);

    return (
        <div>
            <h1>Transactions</h1>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Description</th>
                        <th>Amount</th>
                        <th>Currency</th>
                    </tr>
                </thead>
                <tbody>
                    {transactions.map(transaction => (
                        <tr key={transaction.id}>
                            <td>{new Date(transaction.posted_at).toLocaleDateString()}</td>
                            <td>{transaction.merchant_name}</td>
                            <td>{(transaction.amount_minor / 100).toFixed(2)}</td>
                            <td>{transaction.currency}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default Transactions;