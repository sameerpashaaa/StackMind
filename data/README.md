# Data Directory

This directory is used to store persistent data for the AI Problem Solver application, including:

- Memory storage (vector database files)
- Knowledge graph data
- User session history
- Cached API responses
- Generated visualizations

The contents of this directory are not tracked by version control (added to .gitignore) to avoid committing potentially large data files or sensitive information.

## Structure

```
data/
├── memory/         # Memory system storage
├── knowledge/      # Knowledge graph data
├── sessions/       # User session history
├── cache/          # Cached API responses
└── visualizations/ # Generated visualizations
```

## Notes

- The application will automatically create necessary subdirectories as needed.
- To clear all stored data, you can safely delete the contents of this directory (but not the directory itself).
- Some features may not work properly if the data directory is deleted while the application is running.