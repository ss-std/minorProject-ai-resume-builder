import axios from 'axios';

// This is where my Python FastAPI server is running
const API_BASE_URL = "http://localhost:8000";

export const generateResumeSummary = async (text) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/generate-summary`, {
            raw_text: text
        });
        return response.data.summary;
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
};