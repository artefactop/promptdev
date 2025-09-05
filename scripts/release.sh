#!/bin/bash
# Simple release script using uv's built-in version management

set -e

# Check if version type is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/release.sh <version_type>"
    echo "Version types:"
    echo "  patch  - 0.0.1 -> 0.0.2"
    echo "  minor  - 0.0.1 -> 0.1.0"
    echo "  major  - 0.0.1 -> 1.0.0"
    echo ""
    echo "Or specify exact version:"
    echo "  ./scripts/release.sh 0.0.2"
    exit 1
fi

VERSION_ARG=$1

# Check if it's a version type or exact version
if [[ "$VERSION_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Exact version provided
    echo "Setting version to $VERSION_ARG"
    # We'll need to manually update pyproject.toml for exact versions
    # For now, use uv-version for increment types
    echo "⚠️  For exact versions, manually edit pyproject.toml version field"
    echo "Current version:"
    grep "version" pyproject.toml | head -1
    exit 1
else
    # Version type provided (patch, minor, major)
    echo "Incrementing $VERSION_ARG version..."
    
    # Map common terms to uv-version terms
    case $VERSION_ARG in
        "patch")
            UV_VERSION_TYPE="micro"
            ;;
        "minor")
            UV_VERSION_TYPE="minor"
            ;;
        "major")
            UV_VERSION_TYPE="major"
            ;;
        *)
            echo "❌ Unknown version type: $VERSION_ARG"
            echo "Use: patch, minor, or major"
            exit 1
            ;;
    esac
    
    # Show current version
    echo "Current version:"
    grep "version" pyproject.toml | head -1
    
    # Increment version using uv-version
    echo "Incrementing $UV_VERSION_TYPE version..."
    uvx uv-version increment $UV_VERSION_TYPE
    
    # Show new version
    NEW_VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)
    echo "New version: $NEW_VERSION"
fi

# Get the new version for git operations
NEW_VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)

echo ""
echo "🚀 Creating release v$NEW_VERSION"
echo "======================="

# Git operations
echo "Adding pyproject.toml..."
git add pyproject.toml

echo "Committing version bump..."
git commit -m "bump: version $NEW_VERSION"

echo "Pushing to main..."
git push origin main

echo "Creating GitHub release..."
gh release create v$NEW_VERSION \
    --title "PromptDev v$NEW_VERSION" \
    --notes "Release v$NEW_VERSION" \
    --generate-notes

echo ""
echo "🎉 Release v$NEW_VERSION created!"
echo "The GitHub Action will automatically publish to PyPI."
echo ""
echo "Check the workflow: https://github.com/artefactop/promptdev/actions"
