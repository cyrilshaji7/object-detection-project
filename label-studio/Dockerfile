# Use the official Label Studio image
FROM label-studio/label-studio:latest

# Expose the port Label Studio runs on
EXPOSE 8080

# Set environment variable to enable local file serving
ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=True

# Start Label Studio without opening the browser
CMD ["label-studio", "start", "--no-browser"]
