import React, { useState } from 'react';
import { generateResumeSummary } from '../services/api';

const Home = () => {
    const [inputText, setInputText] = useState("");
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        setLoading(true);
        try {
            const summary = await generateResumeSummary(inputText);
            setResult(summary);
        } catch (err) {
            alert("Check if your Python server is running!");
        }
        setLoading(false);
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
            <h1>IBM AI Resume Builder</h1>
            <p>Paste your Jamia Millia project details below:</p>
            
            <textarea 
                style={{ width: '100%', height: '100px', padding: '10px' }}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="e.g. Worked on SAR data compression..."
            />
            
            <button 
                onClick={handleSubmit} 
                style={{ marginTop: '10px', padding: '10px 20px', cursor: 'pointer' }}
                disabled={loading}
            >
                {loading ? "AI is processing..." : "Generate Professional Summary"}
            </button>

            {result && (
                <div style={{ marginTop: '20px', border: '1px solid #0066cc', padding: '15px' }}>
                    <h3>Result:</h3>
                    <p>{result}</p>
                </div>
            )}
        </div>
    );
};

export default Home;