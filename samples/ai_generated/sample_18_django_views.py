"""Django views and serializers sample."""
from django.db import models
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.utils import timezone
from rest_framework import serializers, viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
import logging

logger = logging.getLogger(__name__)


class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="articles")
    category = models.CharField(max_length=100, default="general")
    published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    view_count = models.IntegerField(default=0)
    tags = models.ManyToManyField("Tag", blank=True, related_name="articles")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "article"
        verbose_name_plural = "articles"

    def __str__(self) -> str:
        return self.title

    def get_absolute_url(self) -> str:
        return f"/articles/{self.pk}/"

    def increment_views(self) -> None:
        self.view_count += 1
        self.save(update_fields=["view_count"])


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    slug = models.SlugField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="comments")
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_approved = models.BooleanField(default=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"Comment by {self.author.username} on {self.article.title}"


class ArticleSerializer(serializers.ModelSerializer):
    author_name = serializers.ReadOnlyField(source="author.username")
    comment_count = serializers.SerializerMethodField()
    tags = serializers.StringRelatedField(many=True, read_only=True)

    class Meta:
        model = Article
        fields = [
            "id", "title", "content", "author", "author_name",
            "category", "published", "created_at", "updated_at",
            "view_count", "comment_count", "tags",
        ]
        read_only_fields = ["author", "created_at", "updated_at", "view_count"]

    def get_comment_count(self, obj) -> int:
        return obj.comments.count()

    def validate_title(self, value: str) -> str:
        if len(value) < 5:
            raise serializers.ValidationError("Title must be at least 5 characters.")
        return value

    def create(self, validated_data: dict) -> Article:
        validated_data["author"] = self.context["request"].user
        return super().create(validated_data)


class CommentSerializer(serializers.ModelSerializer):
    author_name = serializers.ReadOnlyField(source="author.username")

    class Meta:
        model = Comment
        fields = ["id", "article", "author", "author_name", "content", "created_at", "is_approved"]
        read_only_fields = ["author", "created_at"]

    def create(self, validated_data: dict) -> Comment:
        validated_data["author"] = self.context["request"].user
        return super().create(validated_data)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.select_related("author").prefetch_related("tags", "comments")
    serializer_class = ArticleSerializer
    pagination_class = StandardPagination
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        queryset = super().get_queryset()
        category = self.request.query_params.get("category")
        if category:
            queryset = queryset.filter(category=category)
        published = self.request.query_params.get("published")
        if published is not None:
            queryset = queryset.filter(published=published.lower() == "true")
        search = self.request.query_params.get("search")
        if search:
            queryset = queryset.filter(
                models.Q(title__icontains=search) | models.Q(content__icontains=search)
            )
        return queryset

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.increment_views()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)
        logger.info(f"Article created: {serializer.instance.title}")

    def perform_destroy(self, instance):
        logger.warning(f"Article deleted: {instance.title} by {self.request.user}")
        instance.delete()


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def user_articles(request):
    articles = Article.objects.filter(author=request.user).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(articles, request)
    serializer = ArticleSerializer(page, many=True, context={"request": request})
    return paginator.get_paginated_response(serializer.data)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def toggle_publish(request, pk: int):
    article = get_object_or_404(Article, pk=pk, author=request.user)
    article.published = not article.published
    article.save(update_fields=["published"])
    return Response({"published": article.published}, status=status.HTTP_200_OK)


@login_required
def article_list_view(request):
    articles = Article.objects.filter(published=True).select_related("author")
    paginator = Paginator(articles, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "articles/list.html", {"page_obj": page_obj})


@login_required
def article_detail_view(request, pk: int):
    article = get_object_or_404(Article, pk=pk)
    article.increment_views()
    comments = article.comments.filter(is_approved=True).select_related("author")
    context = {"article": article, "comments": comments}
    return render(request, "articles/detail.html", context)
