import asyncio
import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import discord
from groq import APITimeoutError, GroqError

import main


def admin_action(**overrides):
    action = {
        "type": "ADMIN_ACTION",
        "operation": "MODIFY",
        "resource_type": "role",
        "target": "Member",
        "name": None,
        "channel_type": None,
        "color": None,
        "hoist": None,
        "permissions": None,
        "topic": None,
        "category": None,
        "position": None,
        "permission_target": None,
        "allow_permissions": None,
        "deny_permissions": None,
    }
    action.update(overrides)
    return action


class FakeRole:
    def __init__(self, name, *, default=False):
        self.name = name
        self.default = default
        self.managed = False
        self.id = 1 if default else hash(name) & 0xFFFF
        self.position = 0
        self.permissions = discord.Permissions(
            send_messages=True,
            manage_threads=True,
        )

    def is_default(self):
        return self.default

    def __ge__(self, _other):
        return False

    async def edit(self, **kwargs):
        self.edited = kwargs


class FakeRateLimitError(GroqError):
    status_code = 429

    def __init__(self, message="rate limited"):
        super().__init__(message)


class FakeCapacityError(GroqError):
    status_code = 413

    def __init__(self, message="request too large"):
        super().__init__(message)


class FakeModelNotFoundError(GroqError):
    status_code = 404

    def __init__(self, message="model_not_found"):
        super().__init__(message)


class FakeJsonValidationError(GroqError):
    status_code = 400
    body = {
        "error": {
            "type": "invalid_request_error",
            "code": "json_validate_failed",
        }
    }

    def __init__(self):
        super().__init__("json_validate_failed")


class ServerAdminTests(unittest.IsolatedAsyncioTestCase):
    def test_system_prompt_preserves_explicit_names_and_matches_server_style(self):
        self.assertIn("use that exact string in the action", main.SYSTEM_PROMPT)
        self.assertIn("never substitute a merely similar category or channel", main.SYSTEM_PROMPT)
        self.assertIn(
            "Before inventing any category or channel names",
            main.SYSTEM_PROMPT,
        )
        self.assertIn(
            "Explicit user naming and style instructions always take precedence",
            main.SYSTEM_PROMPT,
        )
        self.assertIn("never treat one flag as a complete proxy", main.SYSTEM_PROMPT)
        self.assertIn("send_messages_in_threads", main.SYSTEM_PROMPT)
        self.assertIn("Discord permissions cannot enforce", main.SYSTEM_PROMPT)
        self.assertIn("dependent permission flag individually", main.SYSTEM_PROMPT)
        self.assertIn(
            "strongest theme and emotional\nvibe",
            main.SYSTEM_PROMPT,
        )
        self.assertIn("Any genre or aesthetic is valid", main.SYSTEM_PROMPT)
        self.assertIn("build on them with original imagery", main.SYSTEM_PROMPT)
        self.assertIn("UNIVERSAL AESTHETIC BLUEPRINT", main.SYSTEM_PROMPT)
        self.assertIn("Always apply a clear visual structure", main.SYSTEM_PROMPT)
        self.assertIn("theme-matched emoji and a clean divider\nprefix", main.SYSTEM_PROMPT)
        self.assertIn("︱emoji︱channel-name", main.SYSTEM_PROMPT)
        self.assertIn("・emoji・channel-name", main.SYSTEM_PROMPT)
        self.assertIn("invent or blend directions freely to suit any community", main.SYSTEM_PROMPT)
        self.assertIn("Select distinct but", main.SYSTEM_PROMPT)
        self.assertIn("Use valid\nUnicode and emoji sequences", main.SYSTEM_PROMPT)
        self.assertIn("Explicit user naming and style instructions always take precedence", main.SYSTEM_PROMPT)

    def setUp(self):
        self.everyone = FakeRole("@everyone", default=True)
        self.member = FakeRole("Community Member")
        self.staff = FakeRole("Staff")
        self.text_channel = SimpleNamespace(
            name="📜-rules",
            type=discord.ChannelType.text,
            category=None,
            position=1,
            overwrites={},
            topic="Old topic",
            edit=AsyncMock(),
            set_permissions=AsyncMock(),
            overwrites_for=lambda _role: discord.PermissionOverwrite(),
        )
        self.voice_channel = SimpleNamespace(
            name="Gaming Lounge",
            type=discord.ChannelType.voice,
            category=None,
            position=2,
            overwrites={},
            edit=AsyncMock(),
        )
        self.category = SimpleNamespace(
            name="🎧・VOICE",
            type=discord.ChannelType.category,
            position=0,
            id=20,
            overwrites={},
            edit=AsyncMock(),
        )
        self.guild = SimpleNamespace(
            id=1,
            roles=[self.everyone, self.member, self.staff],
            categories=[self.category],
            text_channels=[self.text_channel],
            voice_channels=[self.voice_channel],
            channels=[self.text_channel, self.voice_channel],
            default_role=self.everyone,
            me=SimpleNamespace(
                guild_permissions=SimpleNamespace(administrator=True),
                top_role=object(),
            ),
            name="Test Server",
            create_category=AsyncMock(),
            create_text_channel=AsyncMock(
                return_value=SimpleNamespace(name="general")
            ),
            create_voice_channel=AsyncMock(
                return_value=SimpleNamespace(name="Lounge")
            ),
            fetch_roles=AsyncMock(side_effect=lambda: list(self.guild.roles)),
            fetch_channels=AsyncMock(
                side_effect=lambda: [
                    *self.guild.channels,
                    *self.guild.categories,
                ]
            ),
        )

    def test_groq_api_key_parser_accepts_csv_and_legacy_single_key(self):
        self.assertEqual(
            main.parse_groq_api_keys(
                {"GROQ_API_KEYS": " key-a, key-b,key-a ,, "}
            ),
            ["key-a", "key-b"],
        )
        self.assertEqual(
            main.parse_groq_api_keys({"GROQ_API_KEY": " legacy-key "}),
            ["legacy-key"],
        )
        self.assertEqual(main.parse_groq_api_keys({}), [])

    async def test_regular_category_creation_passes_an_overwrite_dictionary(self):
        self.guild.categories = []
        result = await main.execute_action(
            self.guild,
            {"type": "CREATE_CATEGORY", "name": "💬・COMMUNITY"},
            "test",
        )

        self.guild.create_category.assert_awaited_once_with(
            "💬・COMMUNITY",
            overwrites={},
            reason="test",
        )
        self.assertIn("Created category", result)

    async def test_regular_text_and_voice_creation_pass_empty_overwrites(self):
        text_action = {
            "type": "CREATE_CHANNEL",
            "name": "general",
            "channel_type": "text",
            "category": None,
            "topic": None,
        }
        voice_action = {
            "type": "CREATE_CHANNEL",
            "name": "Lounge",
            "channel_type": "voice",
            "category": None,
            "topic": None,
        }

        await main.execute_action(self.guild, text_action, "test")
        await main.execute_action(self.guild, voice_action, "test")

        self.guild.create_text_channel.assert_awaited_once_with(
            "general",
            category=None,
            topic=None,
            overwrites={},
            reason="test",
        )
        self.guild.create_voice_channel.assert_awaited_once_with(
            "Lounge",
            category=None,
            overwrites={},
            reason="test",
        )

    async def test_discord_api_call_retries_global_http_429(self):
        error = discord.HTTPException(
            SimpleNamespace(
                status=429,
                headers={},
                reason="Too Many Requests",
            ),
            '{"retry_after": 0.01, "global": true}',
        )
        operation = AsyncMock(side_effect=[error, "complete"])

        with patch("main.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await main._discord_api_call(operation)

        self.assertEqual(result, "complete")
        self.assertEqual(operation.await_count, 2)
        sleep.assert_awaited_once()
        self.assertAlmostEqual(sleep.await_args.args[0], 0.01, places=4)

    async def test_discord_api_call_retries_discord_rate_limited_exception(self):
        operation = AsyncMock(
            side_effect=[discord.RateLimited(0.02), "complete"]
        )

        with patch("main.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await main._discord_api_call(operation)

        self.assertEqual(result, "complete")
        self.assertEqual(operation.await_count, 2)
        sleep.assert_awaited_once_with(0.02)

    async def test_move_all_voice_channels_runs_independent_edits_concurrently(self):
        active = 0
        peak_active = 0

        async def edit_channel(**_kwargs):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0)
            active -= 1

        channels = [
            SimpleNamespace(
                name=f"voice-{index}",
                overwrites={},
                edit=AsyncMock(side_effect=edit_channel),
            )
            for index in range(3)
        ]
        self.guild.voice_channels = channels

        result = await main.move_all_voice_channels(
            self.guild, self.category, "test"
        )

        self.assertEqual(peak_active, 3)
        self.assertIn("Moved 3 of 3", result)
        for channel in channels:
            channel.edit.assert_awaited_once()

    def test_independent_channel_creation_actions_can_batch(self):
        first = {
            "type": "CREATE_CHANNEL",
            "name": "one",
            "channel_type": "text",
            "category": None,
            "topic": None,
        }
        second = {**first, "name": "two"}

        self.assertTrue(
            main._actions_can_run_together(
                self.guild, "CREATE_CHANNEL", [first], second
            )
        )
        self.assertFalse(
            main._actions_can_run_together(
                self.guild, "CREATE_CHANNEL", [first], {**first}
            )
        )

    def test_independent_category_creations_batch_only_without_fuzzy_matches(self):
        first = {
            "type": "CREATE_CATEGORY",
            "name": "project-lab",
        }
        second = {
            "type": "CREATE_CATEGORY",
            "name": "quiet-garden",
        }

        self.assertTrue(
            main._actions_can_run_together(
                self.guild, "CREATE_CATEGORY", [first], second
            )
        )
        self.assertFalse(
            main._actions_can_run_together(
                self.guild,
                "CREATE_CATEGORY",
                [first],
                {**second, "name": "project-lab-support"},
            )
        )

    async def test_delete_uses_fresh_channels_not_stale_guild_cache(self):
        stale_channel = SimpleNamespace(
            id=101,
            name="retired-private-room",
            type=discord.ChannelType.text,
            delete=AsyncMock(),
        )
        current_channel = SimpleNamespace(
            id=101,
            name="fresh-public-room",
            type=discord.ChannelType.text,
            delete=AsyncMock(),
        )
        self.guild.channels = [stale_channel]
        self.guild.fetch_channels.side_effect = None
        self.guild.fetch_channels.return_value = [current_channel]

        with self.assertRaises(main.PlanError):
            await main.execute_action(
                self.guild,
                {
                    "type": "DELETE_CHANNEL",
                    "channel_name": "retired-private-room",
                },
                "test",
            )

        stale_channel.delete.assert_not_awaited()
        current_channel.delete.assert_not_awaited()
        self.guild.fetch_channels.assert_awaited_once()
        self.guild.fetch_roles.assert_awaited_once()

    def test_channel_candidate_lookup_handles_voice_and_generic_resources(self):
        voice, label = main._admin_resource_candidates(
            self.guild, "voice_channel", None
        )
        self.assertEqual(voice, [self.voice_channel])
        self.assertEqual(label, "voice channel")

        channels, label = main._admin_resource_candidates(
            self.guild, "channel", None
        )
        self.assertEqual(channels, [self.text_channel, self.voice_channel])
        self.assertEqual(label, "channel")

    def test_short_fuzzy_match_is_not_allowed_for_deletion(self):
        with self.assertRaises(main.PlanError):
            main._resolve_target(
                [self.member], "mem", "role", destructive=True
            )

        self.assertIs(
            main._resolve_target(
                [self.member], "Community Member", "role", destructive=True
            ),
            self.member,
        )

    async def test_generic_move_can_remove_channel_from_category(self):
        action = admin_action(
            operation="MOVE",
            resource_type="voice_channel",
            target="Gaming Lounge",
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        await main.execute_action(self.guild, validated, "test")

        self.voice_channel.edit.assert_awaited_once()
        self.assertIsNone(
            self.voice_channel.edit.await_args.kwargs["category"]
        )

    async def test_role_position_changes_cannot_cross_bot_highest_role(self):
        bot_top_role = FakeRole("Bot")
        bot_top_role.position = 5
        self.guild.me.top_role = bot_top_role
        self.member.edit = AsyncMock()
        for operation in ("MOVE", "MODIFY"):
            action = admin_action(
                operation=operation,
                resource_type="role",
                target="Community Member",
                position=5,
            )
            validated = main.validate_plan(
                {"actions": [action], "audit_report": []}
            )[0]

            with self.assertRaisesRegex(main.PlanError, "above the bot"):
                await main.execute_action(self.guild, validated, "test")

        self.member.edit.assert_not_awaited()

    async def test_role_permission_delta_preserves_unspecified_permissions(self):
        action = admin_action(
            allow_permissions=["read_message_history"],
            deny_permissions=["manage_threads"],
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        await main.execute_action(self.guild, validated, "test")

        permissions = self.member.edited["permissions"]
        self.assertTrue(permissions.send_messages)
        self.assertTrue(permissions.read_message_history)
        self.assertFalse(permissions.manage_threads)

    async def test_role_creation_refuses_ambiguous_duplicate_name_matches(self):
        duplicate = FakeRole("Moderator")
        first = FakeRole("Moderator")
        self.guild.roles = [self.everyone, first, duplicate]
        action = admin_action(
            operation="CREATE",
            resource_type="role",
            target=None,
            name="Moderator",
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        with self.assertRaisesRegex(main.PlanError, "Multiple roles"):
            await main.execute_action(self.guild, validated, "test")

        self.assertFalse(hasattr(first, "edited"))
        self.assertFalse(hasattr(duplicate, "edited"))

    async def test_create_role_reuse_removes_permissions_not_requested(self):
        action = admin_action(
            operation="CREATE",
            resource_type="role",
            target=None,
            name=self.member.name,
            permissions=["read_message_history"],
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        await main.execute_action(self.guild, validated, "test")

        permissions = self.member.edited["permissions"]
        self.assertTrue(permissions.read_message_history)
        self.assertFalse(permissions.send_messages)
        self.assertFalse(permissions.manage_threads)

    async def test_admin_move_applies_read_only_category_policy(self):
        rules_category = SimpleNamespace(
            id=30,
            name="rules",
            type=discord.ChannelType.category,
            position=0,
            overwrites={},
        )
        rules_category.edit = AsyncMock()
        rules_category.overwrites_for = lambda _role: discord.PermissionOverwrite()
        rules_category.category = None
        self.guild.categories = [rules_category]
        action = admin_action(
            operation="MOVE",
            resource_type="text_channel",
            target=self.text_channel.name,
            category="rules",
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        await main.execute_action(self.guild, validated, "test")

        overwrites = self.text_channel.edit.await_args.kwargs["overwrites"]
        everyone = overwrites[self.everyone]
        self.assertFalse(everyone.send_messages)
        self.assertFalse(everyone.create_public_threads)

    async def test_duplicate_category_reuse_passes_live_state_verification(self):
        categories = [
            SimpleNamespace(
                id=index,
                name="Community",
                position=index,
                type=discord.ChannelType.category,
                overwrites={},
                category=None,
                overwrites_for=lambda _role: discord.PermissionOverwrite(),
            )
            for index in (1, 2)
        ]
        self.guild.categories = categories
        action = admin_action(
            operation="CREATE",
            resource_type="category",
            target=None,
            name="Community",
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        await main.verify_action_state(self.guild, validated)

    async def test_staff_role_hardening_failure_is_reported_as_partial_execution(self):
        staff_category = SimpleNamespace(
            name="🛡️-staff",
            type=discord.ChannelType.category,
            position=0,
            id=21,
            overwrites={},
            edit=AsyncMock(side_effect=RuntimeError("simulated failure")),
        )
        new_role = FakeRole("Staff Moderator")
        self.guild.categories = [staff_category]
        self.guild.channels = []
        self.guild.create_role = AsyncMock(return_value=new_role)
        action = {
            "type": "CREATE_ROLE",
            "name": "Staff Moderator",
            "color": "#5865F2",
            "hoist": True,
            "permissions": [],
        }
        with self.assertRaisesRegex(
            main.PartialExecutionError, "Created role.*could not grant"
        ):
            await main.execute_action(self.guild, action, "test")

    async def test_read_only_channel_denies_thread_creation_and_posting(self):
        overwrites = main._channel_overwrites(
            self.guild,
            "rules",
            None,
            current={self.staff: discord.PermissionOverwrite(send_messages=True)},
        )
        everyone = overwrites[self.everyone]
        self.assertTrue(everyone.view_channel)
        self.assertFalse(everyone.send_messages)
        self.assertFalse(everyone.send_messages_in_threads)
        self.assertFalse(everyone.create_public_threads)
        self.assertFalse(everyone.create_private_threads)
        self.assertFalse(everyone.manage_threads)
        self.assertFalse(everyone.send_tts_messages)
        self.assertFalse(everyone.send_polls)
        self.assertFalse(everyone.send_voice_messages)
        self.assertIn(self.staff, overwrites)

    def test_read_only_overwrites_skip_flags_missing_from_older_discord_py(self):
        supported = frozenset(
            {"view_channel", "read_message_history", "send_messages"}
        )
        with patch.object(main, "VALID_PERMISSIONS", supported):
            overwrites = main._channel_overwrites(
                self.guild, "rules", None
            )

        everyone = overwrites[self.everyone]
        self.assertTrue(everyone.view_channel)
        self.assertTrue(everyone.read_message_history)
        self.assertFalse(everyone.send_messages)

    def test_denying_base_messages_expands_to_thread_send_and_creation_flags(self):
        action = admin_action(
            operation="UPDATE_PERMISSIONS",
            resource_type="channel",
            target="rules",
            permission_target="Community Member",
            deny_permissions=["send_messages"],
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]

        self.assertEqual(
            validated["deny_permissions"],
            [
                "send_messages",
                "send_messages_in_threads",
                "create_public_threads",
                "create_private_threads",
                "manage_threads",
                "send_tts_messages",
                "send_polls",
                "send_voice_messages",
            ],
        )

    async def test_permission_verification_checks_every_expanded_flag(self):
        action = admin_action(
            operation="UPDATE_PERMISSIONS",
            resource_type="channel",
            target="📜-rules",
            permission_target="Community Member",
            deny_permissions=["send_messages"],
        )
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]
        overwrite = discord.PermissionOverwrite(
            send_messages=False,
            send_messages_in_threads=False,
            create_public_threads=False,
            create_private_threads=False,
            manage_threads=False,
            send_tts_messages=False,
            send_polls=False,
            send_voice_messages=False,
        )
        self.text_channel.overwrites_for = lambda _role: overwrite

        await main.verify_action_state(self.guild, validated)

        overwrite.create_private_threads = True
        with self.assertRaisesRegex(main.PlanError, "create_private_threads"):
            await main.verify_action_state(self.guild, validated)

        overwrite.create_private_threads = False
        overwrite.send_voice_messages = True
        with self.assertRaisesRegex(main.PlanError, "send_voice_messages"):
            await main.verify_action_state(self.guild, validated)

    def test_generic_action_rejects_invalid_and_oversized_fields(self):
        invalid = admin_action(operation=[])
        with self.assertRaises(main.PlanError):
            main.validate_plan({"actions": [invalid], "audit_report": []})

        oversized = admin_action(
            operation="MODIFY",
            target="Member",
            name="x" * (main.MAX_RESOURCE_NAME_LENGTH + 1),
        )
        with self.assertRaises(main.PlanError):
            main.validate_plan({"actions": [oversized], "audit_report": []})

    def test_channel_payload_validation_preserves_unicode_but_rejects_invalid_text(self):
        action = {
            "type": "CREATE_CHANNEL",
            "name": "🌌・café 👩‍🚀",
            "channel_type": "text",
            "category": None,
            "topic": "Welcome\nUse this space ✨",
        }
        validated = main.validate_plan(
            {"actions": [action], "audit_report": []}
        )[0]
        self.assertEqual(validated["name"], "🌌・café 👩‍🚀")
        self.assertEqual(validated["topic"], "Welcome\nUse this space ✨")

        for invalid_name in ("invalid\x00name", "invalid\ud800name"):
            invalid = {**action, "name": invalid_name}
            with self.subTest(name=repr(invalid_name)):
                with self.assertRaises(main.PlanError):
                    main.validate_plan({"actions": [invalid], "audit_report": []})

    def test_permission_update_requires_a_real_permission_delta(self):
        action = admin_action(
            operation="UPDATE_PERMISSIONS",
            resource_type="channel",
            target="rules",
            permission_target="Member",
            allow_permissions=[],
            deny_permissions=[],
        )
        with self.assertRaises(main.PlanError):
            main.validate_plan({"actions": [action], "audit_report": []})

    def test_action_journal_persists_history_per_guild(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            journal = main.ActionJournal(database)
            action = {"type": "CREATE_CATEGORY", "name": "community"}
            journal.record(10, 20, action, "SUCCESS: created")

            reloaded = main.ActionJournal(database)
            history = reloaded.recent(10)
            other_guild_history = reloaded.recent(11)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], action)
        self.assertEqual(history[0]["outcome"], "SUCCESS: created")
        self.assertEqual(other_guild_history, [])
        self.assertNotIn("prompt", json.dumps(history))

    def test_planning_context_includes_live_state_and_action_history(self):
        snapshot = {"guild_name": "Test Server", "channels": []}
        history = [{"action": {"type": "CREATE_ROLE"}, "outcome": "SUCCESS"}]

        context = json.loads(
            main._planning_message("Add a staff role", snapshot, history)
        )

        self.assertEqual(context["request"], "Add a staff role")
        self.assertEqual(context["live_guild_snapshot"], snapshot)
        self.assertEqual(context["recent_action_history"], history)

    def test_planning_context_truncates_oversized_inventory_explicitly(self):
        old_limit = main.MAX_CONTEXT_CHARS
        main.MAX_CONTEXT_CHARS = 1800
        snapshot = {
            "guild_name": "Test Server",
            "roles": [],
            "channels": [
                {"name": f"channel-{index}", "topic": "x" * 700}
                for index in range(10)
            ],
            "truncated": {"roles": False, "channels": False},
        }
        try:
            serialized = main._planning_message("organize this", snapshot, [])
        finally:
            main.MAX_CONTEXT_CHARS = old_limit

        context = json.loads(serialized)
        self.assertLessEqual(len(serialized), 1800)
        self.assertTrue(context["live_guild_snapshot"]["truncated"]["channels"])

    def test_rate_limit_error_extracts_minutes_and_seconds(self):
        error = type(
            "RateLimitLikeError",
            (),
            {
                "status_code": 429,
                "response": SimpleNamespace(headers={}),
                "__str__": lambda _self: "Please retry in 3m48s",
            },
        )()

        self.assertTrue(main._is_groq_rate_limit_error(error))
        self.assertEqual(main._retry_wait_time(error), "3 minutes and 48 seconds")

    def test_rate_limit_error_prefers_retry_after_header(self):
        error = type(
            "RateLimitLikeError",
            (),
            {
                "status_code": 429,
                "response": SimpleNamespace(headers={"retry-after": "228"}),
                "__str__": lambda _self: "rate limited",
            },
        )()

        self.assertEqual(main._retry_wait_time(error), "3 minutes and 48 seconds")

    def test_rate_limit_without_duration_has_safe_wait_message(self):
        error = type(
            "RateLimitLikeError",
            (),
            {
                "status_code": 429,
                "response": SimpleNamespace(headers={}),
                "__str__": lambda _self: "rate limited",
            },
        )()

        self.assertEqual(main._retry_wait_time(error), "a little while")

    def test_json_schema_validation_error_is_recognized(self):
        error = FakeJsonValidationError()

        self.assertTrue(main._is_groq_json_validation_error(error))
        self.assertFalse(main._is_groq_json_validation_error(FakeRateLimitError()))

    def test_token_capacity_error_is_recognized_from_api_error_details(self):
        error = SimpleNamespace(
            status_code=400,
            body={
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
            __str__=lambda _self: "context_length_exceeded",
        )

        self.assertTrue(main._is_groq_capacity_error(error))
        self.assertFalse(main._is_groq_capacity_error(FakeRateLimitError()))

    async def test_model_rotation_switches_and_retries_after_rate_limit(self):
        success = object()
        create = AsyncMock(
            side_effect=[FakeRateLimitError("try again in 30s"), success]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        result = await rotation.create_completion(client, messages=[])

        self.assertIs(result, success)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )
        self.assertEqual(rotation.active_model, "llama-3.3-70b-versatile")
        self.assertEqual(
            rotation.models,
            ("openai/gpt-oss-120b", "llama-3.3-70b-versatile"),
        )

    async def test_model_rotation_trims_context_after_capacity_error(self):
        success = object()
        create = AsyncMock(
            side_effect=[FakeCapacityError(), FakeRateLimitError(), success]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        context = {
            "request": "organize channels",
            "live_guild_snapshot": {
                "channels": [
                    {"name": f"channel-{index}", "overwrites": [{"deny": ["x"]}]}
                    for index in range(8)
                ],
                "roles": [{"name": f"role-{index}"} for index in range(8)],
                "categories": [{"name": f"category-{index}"} for index in range(8)],
                "truncated": {"channels": False, "roles": False},
                "audit_facts": {
                    "role_hierarchy": [{"name": f"role-{index}"} for index in range(8)]
                },
            },
            "recent_action_history": [{"action": str(index)} for index in range(8)],
        }
        messages = [
            {"role": "system", "content": "instructions"},
            {
                "role": "user",
                "content": json.dumps(context, separators=(",", ":")),
            },
        ]
        rotation = main.GroqModelRotation()

        result = await rotation.create_completion(client, messages=messages)

        self.assertIs(result, success)
        self.assertEqual(create.await_count, 3)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-120b",
                "llama-3.3-70b-versatile",
            ],
        )
        original_content = create.await_args_list[0].kwargs["messages"][1]["content"]
        adapted_content = create.await_args_list[1].kwargs["messages"][1]["content"]
        self.assertGreater(len(original_content), len(adapted_content))
        self.assertEqual(
            adapted_content,
            create.await_args_list[2].kwargs["messages"][1]["content"],
        )
        adapted_context = json.loads(adapted_content)
        self.assertTrue(adapted_context["live_guild_snapshot"]["truncated"]["channels"])
        self.assertTrue(adapted_context["live_guild_snapshot"]["truncated"]["overwrites"])
        self.assertNotIn(
            "overwrites", adapted_context["live_guild_snapshot"]["channels"][0]
        )
        self.assertTrue(
            adapted_context["live_guild_snapshot"]["channels"][0][
                "overwrites_truncated"
            ]
        )

    async def test_model_rotation_retries_network_timeouts_on_fallback_model(self):
        success = object()
        create = AsyncMock(
            side_effect=[
                APITimeoutError(request=SimpleNamespace()),
                success,
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = await main.GroqModelRotation().create_completion(client, messages=[])

        self.assertIs(result, success)
        self.assertEqual(create.await_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )

    async def test_six_key_rotation_exhausts_both_models_before_returning_rate_limit(self):
        attempts = []
        clients = []
        for key_index in range(6):
            async def rate_limited(*, model, key_index=key_index, **_kwargs):
                attempts.append((model, key_index))
                raise FakeRateLimitError("rate limited")

            clients.append(
                SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(
                            create=AsyncMock(side_effect=rate_limited)
                        )
                    )
                )
            )

        with self.assertRaises(FakeRateLimitError):
            await main.GroqModelRotation().create_completion(clients, messages=[])

        self.assertEqual(
            attempts,
            [
                (model, key_index)
                for model in main.MODEL_ROTATION
                for key_index in range(6)
            ],
        )

    async def test_model_rotation_alternates_on_repeated_rate_limits_boundedly(self):
        create = AsyncMock(
            side_effect=[
                FakeRateLimitError("primary limited"),
                FakeRateLimitError("secondary limited"),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        with self.assertRaisesRegex(FakeRateLimitError, "secondary limited"):
            await rotation.create_completion(client, messages=[])

        self.assertEqual(create.await_count, main.MAX_MODEL_ATTEMPTS)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )
        self.assertEqual(rotation.active_model, "llama-3.3-70b-versatile")

    async def test_rate_limit_failover_tries_full_model_key_matrix(self):
        clients = []
        create_mocks = []
        attempts = []
        for key_index in range(2):
            async def limited(*, model, key_index=key_index, **_kwargs):
                attempts.append((model, key_index))
                raise FakeRateLimitError("limited")

            create = AsyncMock(side_effect=limited)
            create_mocks.append(create)
            clients.append(
                SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=create)
                    )
                )
            )
        rotation = main.GroqModelRotation()

        with self.assertRaises(FakeRateLimitError):
            await rotation.create_completion(clients, messages=[])

        self.assertEqual(
            attempts,
            [
                ("openai/gpt-oss-120b", 0),
                ("openai/gpt-oss-120b", 1),
                ("llama-3.3-70b-versatile", 0),
                ("llama-3.3-70b-versatile", 1),
            ],
        )
        self.assertEqual(
            [
                [call.kwargs["model"] for call in mock.await_args_list]
                for mock in create_mocks
            ],
            [
                ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
                ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
            ],
        )
        self.assertEqual(sum(mock.await_count for mock in create_mocks), 4)

    async def test_successful_key_is_reused_on_the_next_request(self):
        clients = []
        create_mocks = []
        for outcomes in (
            [FakeRateLimitError("first key limited")],
            [object(), object()],
        ):
            create = AsyncMock(side_effect=outcomes)
            create_mocks.append(create)
            clients.append(
                SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=create)
                    )
                )
            )
        rotation = main.GroqModelRotation()

        await rotation.create_completion(clients, messages=[])
        await rotation.create_completion(clients, messages=[])

        self.assertEqual(rotation.active_key_index, 1)
        self.assertEqual(
            create_mocks[1].await_args_list[0].kwargs["model"],
            "openai/gpt-oss-120b",
        )
        self.assertEqual(create_mocks[1].await_count, 2)

    async def test_rate_limit_message_is_not_selected_for_mixed_matrix_errors(self):
        clients = []
        for errors in (
            [
                FakeRateLimitError("primary key one"),
                FakeModelNotFoundError("missing on first key"),
            ],
            [
                FakeModelNotFoundError("missing on first key"),
                FakeRateLimitError("secondary key two"),
            ],
        ):
            create = AsyncMock(side_effect=errors)
            clients.append(
                SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=create)
                    )
                )
            )
        rotation = main.GroqModelRotation()

        with self.assertRaises(FakeModelNotFoundError):
            await rotation.create_completion(clients, messages=[])

    async def test_model_rotation_does_not_fallback_for_non_429_errors(self):
        failure = RuntimeError("invalid API key")
        create = AsyncMock(side_effect=failure)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        with self.assertRaisesRegex(RuntimeError, "invalid API key"):
            await rotation.create_completion(client, messages=[])

        create.assert_awaited_once()
        self.assertEqual(rotation.active_model, "openai/gpt-oss-120b")

    async def test_model_rotation_falls_back_when_active_model_is_unavailable(self):
        fallback_error = FakeModelNotFoundError()
        success = object()
        create = AsyncMock(
            side_effect=[fallback_error, success]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        result = await rotation.create_completion(client, messages=[])

        self.assertIs(result, success)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )
        self.assertEqual(rotation.active_model, "llama-3.3-70b-versatile")

    async def test_model_rotation_retries_json_schema_validation_errors(self):
        success = object()
        create = AsyncMock(
            side_effect=[FakeJsonValidationError(), success]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        result = await rotation.create_completion(client, messages=[])

        self.assertIs(result, success)
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )

    async def test_both_models_json_validation_failures_return_clean_plan_error(self):
        create = AsyncMock(
            side_effect=[FakeJsonValidationError(), FakeJsonValidationError()]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rotation = main.GroqModelRotation()

        with self.assertRaisesRegex(
            main.PlanError, "No server changes were made"
        ):
            await rotation.create_completion(client, messages=[])

        self.assertEqual(create.await_count, main.MAX_MODEL_ATTEMPTS)

    async def test_model_rotation_retries_malformed_generation(self):
        invalid_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="not valid JSON")
                )
            ]
        )
        valid_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"actions":[],"audit_report":["No changes needed."]}'
                    )
                )
            ]
        )
        create = AsyncMock(side_effect=[invalid_completion, valid_completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[],
            default_role=None,
            channels=[],
            categories=[],
            me=None,
            fetch_roles=AsyncMock(return_value=[]),
            fetch_channels=AsyncMock(return_value=[]),
        )
        rotation = main.GroqModelRotation()

        result = await main.generate_plan(
            client, "organize channels", guild, [], rotation
        )

        self.assertEqual(result, ([], ["No changes needed."]))
        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        )

    async def test_model_rotation_reports_clean_error_if_both_generations_are_invalid(self):
        invalid_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"actions": []}')
                )
            ]
        )
        create = AsyncMock(
            side_effect=[invalid_completion, invalid_completion]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[],
            default_role=None,
            channels=[],
            categories=[],
            me=None,
            fetch_roles=AsyncMock(return_value=[]),
            fetch_channels=AsyncMock(return_value=[]),
        )

        with self.assertRaisesRegex(main.PlanError, "No server changes were made"):
            await main.generate_plan(
                client, "organize channels", guild, [], main.GroqModelRotation()
            )

        self.assertEqual(create.await_count, 2)

    async def test_empty_generation_retries_on_alternate_model(self):
        empty_completion = SimpleNamespace(choices=[])
        valid_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"actions":[],"audit_report":["No changes needed."]}'
                    )
                )
            ]
        )
        create = AsyncMock(side_effect=[empty_completion, valid_completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[],
            default_role=None,
            channels=[],
            categories=[],
            me=None,
            fetch_roles=AsyncMock(return_value=[]),
            fetch_channels=AsyncMock(return_value=[]),
        )

        result = await main.generate_plan(
            client,
            "organize channels",
            guild,
            [],
            main.GroqModelRotation(),
        )

        self.assertEqual(result, ([], ["No changes needed."]))
        self.assertEqual(create.await_count, 2)

    def test_groq_client_disables_sdk_retries_for_immediate_model_fallback(self):
        with patch("main.AsyncGroq") as client_factory:
            main.create_groq_client("test-key")

        client_factory.assert_called_once_with(
            api_key="test-key",
            timeout=60.0,
            max_retries=0,
        )

    async def test_generate_plan_sends_live_snapshot_to_groq(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content='{"actions":[],"audit_report":["No changes needed."]}'
                                    )
                                )
                            ]
                        )
                    )
                )
            )
        )

        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[],
            default_role=None,
            channels=[],
            categories=[],
            me=None,
            fetch_roles=AsyncMock(return_value=[]),
            fetch_channels=AsyncMock(return_value=[]),
        )
        rotation = main.GroqModelRotation()
        result = await main.generate_plan(
            client, "organize channels", guild, [], rotation
        )

        self.assertEqual(result, ([], ["No changes needed."]))
        user_message = client.chat.completions.create.await_args.kwargs["messages"][1]
        self.assertEqual(
            client.chat.completions.create.await_args.kwargs["model"],
            "openai/gpt-oss-120b",
        )
        context = json.loads(user_message["content"])
        self.assertEqual(context["live_guild_snapshot"]["guild_name"], "Test Server")
        self.assertTrue(
            context["live_guild_snapshot"]["audit_facts"][
                "inventory_fetched_from_discord_api"
            ]
        )
        guild.fetch_roles.assert_awaited_once()
        guild.fetch_channels.assert_awaited_once()

    def test_guild_snapshot_includes_current_roles_categories_and_channels(self):
        role = FakeRole("Member")
        category = SimpleNamespace(id=30, name="Community", position=0)
        channel = SimpleNamespace(
            id=40,
            name="general",
            type=SimpleNamespace(value="text"),
            position=1,
            category=category,
            overwrites={},
        )
        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[self.everyone, role],
            default_role=self.everyone,
            channels=[channel],
            categories=[category],
            me=None,
        )

        snapshot = main.guild_snapshot(guild)

        self.assertEqual(snapshot["roles"][0]["name"], "@everyone")
        self.assertEqual(snapshot["categories"][0]["name"], "Community")
        self.assertEqual(snapshot["channels"][0]["name"], "general")
        self.assertEqual(snapshot["channels"][0]["category"], "Community")
        self.assertEqual(snapshot["permission_inventory"], sorted(main.VALID_PERMISSIONS))
        self.assertEqual(snapshot["audit_facts"]["empty_categories"], [])
        self.assertFalse(
            snapshot["audit_facts"]["channel_message_history_inspected"]
        )

    def test_guild_snapshot_finds_duplicate_role_names_and_empty_categories(self):
        category = SimpleNamespace(id=30, name="Empty", position=1)
        guild = SimpleNamespace(
            id=1,
            name="Test Server",
            roles=[self.everyone, FakeRole("Moderator"), FakeRole("moderator")],
            default_role=self.everyone,
            channels=[],
            categories=[category],
            me=SimpleNamespace(
                top_role=SimpleNamespace(id=999, position=10)
            ),
        )

        audit_facts = main.guild_snapshot(guild)["audit_facts"]

        self.assertEqual(
            audit_facts["duplicate_role_name_groups"][0]["name"], "Moderator"
        )
        self.assertEqual(audit_facts["empty_categories"], ["Empty"])
        self.assertEqual(audit_facts["bot_highest_role_position"], 10)
        self.assertFalse(audit_facts["role_membership_counts_available"])

    def test_plan_requires_a_bounded_audit_report(self):
        with self.assertRaises(main.PlanError):
            main.validate_plan({"actions": []})
        with self.assertRaises(main.PlanError):
            main.validate_plan(
                {"actions": [], "audit_report": ["x" * 1001]}
            )


if __name__ == "__main__":
    unittest.main()
